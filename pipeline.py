from google.colab import drive
drive.mount('/content/drive')
# %cd ./drive/MyDrive/refVRAG

import os

if not os.path.exists("LLaVA-NeXT"):
    !git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
# %cd LLaVA-NeXT

!pip install "numpy<2" easyocr --quiet --force-reinstall

import numpy as np
print("NumPy version:", np.__version__)

!pip install --upgrade pip setuptools wheel
!pip install spacy-experimental
!pip install faiss-cpu easyocr ffmpeg-python
!pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.39.0 decord av git+https://github.com/mlfoundations/open_clip.git
!python -m spacy download en_core_web_sm

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    CLIPProcessor,
    CLIPModel,
    DetrForObjectDetection,
    DetrImageProcessor,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from decord import VideoReader, cpu
import numpy as np
import json
from tqdm import tqdm
import os
import easyocr
import re
import ast
import socket
import pickle
import ffmpeg, torchaudio
import spacy

# Commented out IPython magic to ensure Python compatibility.
# %cd ../

from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
from tools.scene_graph import generate_scene_graph_description

LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda"

import spacy
try:
    nlp = spacy.load("en_core_web_sm")

    print("Standard spaCy pipeline (en_core_web_sm) loaded for rule-based analysis.")

except Exception as e:
    print(f"Error loading base spaCy model: {e}")

max_frames_num = 32

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14-336",
    torch_dtype=torch.float16,
    device_map="auto",
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large",
    torch_dtype=torch.float16,
    device_map="auto",
)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")

DETR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DETR_DEVICE)
detr_model.eval()

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

def inference_retriever(prompt_text, max_new_tokens: int = 64):


    messages = [
        {
            "role": "system",
            "content": (
                "You are a specialized expert in video question answering. "
                "You must follow the user instructions exactly. "
                "When answering multiple-choice questions, your final output "
                "should be the letter of the correct option followed by a "
                "period, space, and the full text of that option (e.g., 'A. Texting'). "
                "Do not include any extraneous words, explanations, or quotes."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if text in response:
        response = response.replace(text, "").strip()

    return response.strip()

def resolve_coreference_rule_based(doc, timed_segments):
    antecedents = {}
    pronoun_indices = []

    for token in doc:
        if token.ent_type_ in ["PERSON", "ORG", "LOC"] or token.pos_ in ["NOUN", "PROPN"]:
            antecedents[token.text.lower()] = token.i
        elif token.tag_ in ["PRP", "PRP$"]:
            pronoun_indices.append(token.i)

    token_list = [token.text_with_ws for token in doc]

    for p_idx in pronoun_indices:
        pronoun_token = doc[p_idx]

        is_singular = pronoun_token.text.lower() in ["he", "she", "it", "him", "her"]
        is_plural = pronoun_token.text.lower() in ["they", "them", "we"]

        best_match = None
        best_index = -1

        for e_text, e_idx in antecedents.items():
            if e_idx < p_idx:
                if e_idx > best_index:
                    best_index = e_idx
                    best_match = doc[e_idx].text

        if best_match:
            token_list[p_idx] = best_match + pronoun_token.whitespace_

    resolved_text = "".join(token_list)

    timed_output = []

    resolved_lines = resolved_text.split('\n')

    for i, (time, original_text) in enumerate(timed_segments):
        if i < len(resolved_lines):
            if resolved_lines[i].startswith(f"({time:.2f}s)"):
                 timed_output.append(resolved_lines[i])
            else:
                 timed_output.append(f"({time:.2f}s) [RESOLVED]: {resolved_lines[i]}")

    return "\n".join(timed_output)

def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    vr = VideoReader(video_path, ctx=cpu(),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()

    fps_rate = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps_rate)]

    if len(frame_idx) > max_frames_num or force_sample:
        uniform_sampled_indices = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_indices.tolist()

    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time_str, frame_time, video_time

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()

def chunk_audio(audio_path, chunk_length_s=30):
    import librosa
    import numpy as np

    speech, sr = librosa.load(audio_path, sr=16000)

    samples_per_chunk = int(chunk_length_s * sr)

    chunks = []
    for i in range(0, len(speech), samples_per_chunk):
        chunk = speech[i : i + samples_per_chunk]
        if chunk.size == 0:
            continue
        chunks.append(chunk)

    return chunks

def get_asr_docs(video_path, audio_path, chunk_length_s: float = 30.0):
    import torch

    full_timed_transcription = []

    try:
        extract_audio(video_path, audio_path)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return full_timed_transcription

    audio_chunks = chunk_audio(audio_path, chunk_length_s=chunk_length_s)
    if not audio_chunks:
        return full_timed_transcription

    all_input_features = []
    chunk_start_times = []

    for i, chunk in enumerate(audio_chunks):
        start_time = i * chunk_length_s
        chunk_start_times.append(start_time)

        if isinstance(chunk, torch.Tensor):
            chunk_np = chunk.cpu().numpy()
        else:
            chunk_np = chunk

        inputs = whisper_processor(
            chunk_np,
            sampling_rate=16000,
            return_tensors="pt",
        )
        all_input_features.append(inputs.input_features)

    input_features = torch.cat(all_input_features, dim=0).to(
        whisper_model.device,
        dtype=whisper_model.dtype,
    )

    with torch.no_grad():
        generated_ids = whisper_model.generate(
            input_features,
            language="en",
            task="transcribe",
        )

    texts = whisper_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    for start_time, text in zip(chunk_start_times, texts):
        text = text.strip()
        if not text:
            continue
        full_timed_transcription.append((start_time, f"[ASR]: {text}"))

    return full_timed_transcription

def windowed_asr_docs(timed_asr_docs, window_size: float = 60.0):
    if not timed_asr_docs:
        return []

    timed_asr_docs = sorted(timed_asr_docs, key=lambda x: x[0])
    merged = []
    current_start = timed_asr_docs[0][0]
    current_end = current_start + window_size
    buffer_texts = []

    for ts, text in timed_asr_docs:
        if ts <= current_end:
            buffer_texts.append(text)
        else:
            merged.append(
                (current_start, f"[ASR-WIN {current_start:.2f}-{current_end:.2f}s]: " + " ".join(buffer_texts))
            )
            buffer_texts = [text]
            current_start = ts
            current_end = ts + window_size

    if buffer_texts:
        merged.append(
            (current_start, f"[ASR-WIN {current_start:.2f}-{current_end:.2f}s]: " + " ".join(buffer_texts))
        )

    return merged

import easyocr
reader_ocr = easyocr.Reader(['en'], gpu=True)

def get_ocr_docs(frames, frame_times):
    if frames is None or len(frames) == 0:
        return []

    text_set = set()
    timed_ocr_docs = []

    for i, img in enumerate(frames):
        if i >= len(frame_times):
            break

        frame_timestamp = frame_times[i]

        try:
            ocr_results = reader_ocr.readtext(img)
        except Exception as e:
            print(f"OCR failed on frame {i}, skipping:", e)
            continue

        current_frame_text = []
        for result in ocr_results:
            if len(result) < 3:
                continue
            text = result[1]
            confidence = result[2]
            if confidence > 0.6 and text not in text_set:
                current_frame_text.append(text)
                text_set.add(text)

        if current_frame_text:
            det_info = "; ".join(current_frame_text)
            timed_ocr_docs.append((frame_timestamp, f"[OCR]: {det_info}"))

    return timed_ocr_docs

def need_ocr(question: str) -> bool:
    q = question.lower()
    keywords = [
        "read", "text", "sign", "written", "word", "label",
        "number", "date", "percentage", "price", "year"
    ]
    return any(k in q for k in keywords)


def need_det(question: str) -> bool:
    q = question.lower()
    if "how many" in q:
        return True
    object_keywords = [
        "man", "woman", "person", "people", "kid", "child", "dog", "cat",
        "car", "ball", "shirt", "hat", "holding", "wearing", "color",
        "left", "right", "behind", "front", "in front of"
    ]
    return any(k in q for k in object_keywords)

def get_det_docs(frames, prompt):
    if len(frames) == 0:
        return []

    requested = [p.lower().strip() for p in prompt if isinstance(p, str)]
    requested = [r for r in requested if r]

    from PIL import Image
    import numpy as np
    import torch

    pil_frames = []
    target_sizes = []

    for f in frames:
        if isinstance(f, np.ndarray):
            img = Image.fromarray(f)
        else:
            img = f
        pil_frames.append(img)
        w, h = img.size
        target_sizes.append([h, w])

    inputs = detr_processor(images=pil_frames, return_tensors="pt")
    inputs = {k: v.to(DETR_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = detr_model(**inputs)

    target_sizes = torch.tensor(target_sizes, device=DETR_DEVICE)
    results = detr_processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.5,
    )

    det_docs = []
    for res in results:
        frame_det = []
        scores = res["scores"].detach().cpu().numpy()
        labels = res["labels"].detach().cpu().numpy()
        boxes = res["boxes"].detach().cpu().numpy()

        for score, label_id, box in zip(scores, labels, boxes):
            label = detr_model.config.id2label[int(label_id)].lower()

            if requested and all(r not in label for r in requested):
                continue

            frame_det.append({
                "label": label,
                "bbox": box.tolist(),
                "score": float(score),
            })

        det_docs.append(frame_det)

    return det_docs

from collections import Counter

def det_preprocess(det_docs, det_top_idx, frame_times, location, relation, number):
    timed_scene_descriptions = []

    if len(det_docs) != len(det_top_idx):
        return []

    for i, detections in enumerate(det_docs):
        frame_index = det_top_idx[i]
        frame_timestamp = frame_times[frame_index]

        if not detections:
            continue

        objects = []
        for obj_id, det in enumerate(detections):
            label = det.get("label")
            bbox = det.get("bbox")
            if label is None or bbox is None:
                continue
            objects.append({"id": obj_id, "label": label, "bbox": bbox})

        if not objects:
            continue

        scene_description = generate_scene_graph_description(
            objects,
            location,
            relation,
            number,
        )

        if scene_description:
            timed_scene_descriptions.append(
                (frame_timestamp, f"[SCENE]: Frame {frame_index+1}: {scene_description}")
            )

    return timed_scene_descriptions

!pip install ultralytics --quiet

DETR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DETR_DEVICE)
detr_model.eval()

rag_threshold = 0.2
clip_threshold = 0.3
beta = 3.0


USE_OCR = True
USE_ASR = True
USE_DET = True
print(f"---------------OCR{rag_threshold}: {USE_OCR}-----------------")
print(f"---------------ASR{rag_threshold}: {USE_ASR}-----------------")
print(f"---------------DET{beta}-{clip_threshold}: {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")


video_path = "/content/drive/MyDrive/refVRAG/vids/001.mp4"
question = "What has come a long way since they started? A. Cheating B. Texting  C. Decoration D. Dancing"


frames, frame_time_str, frame_times, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
raw_video = [f for f in frames]

if USE_DET:
    video_tensor = []
    for frame in raw_video:
        processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
        video_tensor.append(processed.squeeze(0))
    video_tensor = torch.stack(video_tensor, dim=0)

retrieve_pmt_0 = "Question: " + question
retrieve_pmt_0 += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
retrieve_pmt_0 += "Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format."

json_request = inference_retriever(retrieve_pmt_0)

try:
    plan = json.loads(json_request)
except Exception as e:
    print("Warning: could not parse retrieval plan JSON, falling back:", e)
    plan = {}

request_det = plan.get("DET", []) or []
request_type = plan.get("TYPE", []) or []

request_det = [str(x) for x in request_det]
request_type = [str(x).lower() for x in request_type]

L = any("loc" in t for t in request_type)
R = any("rel" in t for t in request_type)
N = any("num" in t or "count" in t for t in request_type)
if not (L or R or N):
    N = True

query = [question]
timed_scene_docs = []
timed_ocr_docs = []
timed_asr_docs = []

use_det = USE_DET and need_det(question)

if use_det and len(frames) > 0:
    import numpy as np
    num_det_frames = min(8, len(frames))
    det_top_idx = np.linspace(0, len(frames) - 1, num_det_frames, dtype=int).tolist()

    det_frames = [frames[i] for i in det_top_idx]

    det_docs = get_det_docs(det_frames, request_det)
    timed_scene_docs = det_preprocess(
        det_docs,
        det_top_idx,
        frame_times,
        location=L,
        relation=R,
        number=N,
    )
else:
    timed_scene_docs = []



if USE_OCR and need_ocr(question):
    timed_ocr_docs = get_ocr_docs(frames, frame_times)
else:
    timed_ocr_docs = []


if USE_ASR:
    audio_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".wav")
    timed_asr_docs_raw = get_asr_docs(video_path, audio_path)
    timed_asr_docs = windowed_asr_docs(timed_asr_docs_raw, window_size=60.0)
else:
    timed_asr_docs = []


fused_timed_docs = []
fused_timed_docs.extend(timed_scene_docs)
fused_timed_docs.extend(timed_ocr_docs)
fused_timed_docs.extend(timed_asr_docs)

fused_timed_docs.sort(key=lambda x: x[0])

combined_text_for_coref = "\n".join([f"({time:.2f}s) {text}" for time, text in fused_timed_docs])

resolved_text = ""
if combined_text_for_coref:
    doc = nlp(combined_text_for_coref)
    resolved_text = resolve_coreference_rule_based(doc, fused_timed_docs)

retrieved_context = ""
if resolved_text:
    resolved_docs_total = [resolved_text]

    request_asr = plan.get("ASR")
    asr_query = query.copy()
    if request_asr:
       asr_query = [request_asr]

    fused_retrieved_docs, _ = retrieve_documents_with_dynamic(
        resolved_docs_total, asr_query, threshold=rag_threshold)

    retrieved_context = " ".join(fused_retrieved_docs)



qs = ""

qs += "TASK: Answer the following multiple-choice question based ONLY on the provided Contextualized Scene Description (CSD). Output the letter of the correct option followed by a period, space, and the full text of that option (e.g., 'A. Texting'). Do not include any extraneous words, explanations, or quotes.\n\n"

if retrieved_context:
    qs += "CSD (Contextualized Scene Description):\n"
    qs += retrieved_context
    qs += "\n\n"
else:
     qs += "CSD (Contextualized Scene Description): None available.\n\n"

qs += "QUESTION: " + question


res = inference_retriever(qs)
print(res)

print("len(timed_scene_docs) =", len(timed_scene_docs))
print("len(timed_ocr_docs)   =", len(timed_ocr_docs))
print("len(timed_asr_docs)   =", len(timed_asr_docs))

print("Sample scene:", timed_scene_docs[:2])
print("Sample OCR:", timed_ocr_docs[:2])
print("Sample ASR:", timed_asr_docs[:2])

"""## Evaluation

"""

def generate_answer_with_refvrag(video_path, question):
  rag_threshold = 0.2
  clip_threshold = 0.3
  beta = 3.0

  USE_OCR = True
  USE_ASR = True
  USE_DET = True
  print(f"---------------OCR{rag_threshold}: {USE_OCR}-----------------")
  print(f"---------------ASR{rag_threshold}: {USE_ASR}-----------------")
  print(f"---------------DET{beta}-{clip_threshold}: {USE_DET}-----------------")
  print(f"---------------Frames: {max_frames_num}-----------------")


  frames, frame_time_str, frame_times, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
  raw_video = [f for f in frames]

  if USE_DET:
      video_tensor = []
      for frame in raw_video:
          processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
          video_tensor.append(processed.squeeze(0))
      video_tensor = torch.stack(video_tensor, dim=0)

  retrieve_pmt_0 = "Question: " + question
  retrieve_pmt_0 += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
  retrieve_pmt_0 += "Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format."

  json_request = inference_retriever(retrieve_pmt_0)

  try:
      parts = json_request.split("assistant", 1)
      extracted_answer = parts[1].strip()
      plan = json.loads(extracted_answer)
  except Exception as e:
      print("Warning: could not parse retrieval plan JSON, falling back:", e)
      plan = {}

  request_det = plan.get("DET", []) or []
  request_type = plan.get("TYPE", []) or []

  request_det = [str(x) for x in request_det]
  request_type = [str(x).lower() for x in request_type]

  L = any("loc" in t for t in request_type)
  R = any("rel" in t for t in request_type)
  N = any("num" in t or "count" in t for t in request_type)
  if not (L or R or N):
      N = True

  query = [question]
  timed_scene_docs = []
  timed_ocr_docs = []
  timed_asr_docs = []

  use_det = USE_DET and need_det(question)

  if use_det and len(frames) > 0:
      import numpy as np
      num_det_frames = min(8, len(frames))
      det_top_idx = np.linspace(0, len(frames) - 1, num_det_frames, dtype=int).tolist()

      det_frames = [frames[i] for i in det_top_idx]

      det_docs = get_det_docs(det_frames, request_det)
      timed_scene_docs = det_preprocess(
          det_docs,
          det_top_idx,
          frame_times,
          location=L,
          relation=R,
          number=N,
      )
  else:
      timed_scene_docs = []

  if USE_OCR and need_ocr(question):
      timed_ocr_docs = get_ocr_docs(frames, frame_times)
  else:
      timed_ocr_docs = []

  if USE_ASR:
      audio_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".wav")
      timed_asr_docs_raw = get_asr_docs(video_path, audio_path)
      timed_asr_docs = windowed_asr_docs(timed_asr_docs_raw, window_size=60.0)
  else:
      timed_asr_docs = []


  fused_timed_docs = []
  fused_timed_docs.extend(timed_scene_docs)
  fused_timed_docs.extend(timed_ocr_docs)
  fused_timed_docs.extend(timed_asr_docs)

  fused_timed_docs.sort(key=lambda x: x[0])

  combined_text_for_coref = "\n".join([f"({time:.2f}s) {text}" for time, text in fused_timed_docs])

  resolved_text = ""
  if combined_text_for_coref:
      doc = nlp(combined_text_for_coref)
      resolved_text = resolve_coreference_rule_based(doc, fused_timed_docs)

  retrieved_context = ""
  if resolved_text:
      resolved_docs_total = [resolved_text]

      request_asr = plan.get("ASR")
      asr_query = query.copy()
      if request_asr:
        asr_query = [request_asr]

      fused_retrieved_docs, _ = retrieve_documents_with_dynamic(
          resolved_docs_total, asr_query, threshold=rag_threshold)

      retrieved_context = " ".join(fused_retrieved_docs)

  qs = ""
  qs += "TASK: Answer the following multiple-choice question based ONLY on the provided Contextualized Scene Description (CSD). Output the letter of the correct option followed by a period, space, and the full text of that option (e.g., 'A. Texting'). Do not include any extraneous words, explanations, or quotes.\n\n"

  if retrieved_context:
      qs += "CSD (Contextualized Scene Description):\n"
      qs += retrieved_context
      qs += "\n\n"
  else:
      qs += "CSD (Contextualized Scene Description): None available.\n\n"
  qs += "QUESTION: " + question

  print(qs)
  res = inference_retriever(qs)

  return res

!pip install evaluate
!pip install rouge_score

from datasets import load_dataset
import os

dataset = load_dataset("lmms-lab/Video-MME")
vid_path_folder = "/content/drive/MyDrive/refVRAG/vids/"
all_files = os.listdir(vid_path_folder)
files = [f for f in all_files if f.endswith(".mp4")]
vid_ids = [os.path.splitext(f)[0] for f in files]
dataset = dataset.filter(lambda x: x['video_id'] in vid_ids)



import evaluate
import numpy as np

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

all_bleu_scores = []
all_rouge_scores = []
count_retrieved_correct = 0
total_count = 0

for i in range(len(dataset["test"])):
    sample = dataset["test"][i]
    query = sample["question"] + ", ".join(sample["options"])
    reference = next(opt for opt in sample["options"] if opt.startswith(sample["answer"]))

    vid_path = f"/content/drive/MyDrive/refVRAG/vids/{sample['video_id']}.mp4"

    predicted = generate_answer_with_refvrag(vid_path, query)
    parts = predicted.split("assistant", 1)
    extracted_answer = parts[1].strip()

    # print(f"que: {sample['question']} \n ++++++ ans: {reference} \n +++++++ predicted: {predicted} \n +++++++ predicted: {extracted_answer} \n\n")
    predicted = extracted_answer
    print("================================================")

    # Compute BLEU
    bleu_score = bleu.compute(predictions=[predicted], references=[[reference]])["bleu"]

    # Compute ROUGE-L
    rouge_score = rouge.compute(predictions=[predicted], references=[reference])["rougeL"]
    total_count += 1
    if predicted == reference:
      count_retrieved_correct += 1

    all_bleu_scores.append(bleu_score)
    all_rouge_scores.append(rouge_score)

# Compute average scores
mean_bleu = np.mean(all_bleu_scores)
mean_rouge = np.mean(all_rouge_scores)

mean_bleu = np.mean(all_bleu_scores)
mean_rouge = np.mean(all_rouge_scores)

print(f"Average BLEU-4: {mean_bleu:.4f}")
print(f"Average ROUGE-L: {mean_rouge:.4f}")
print(f"Precision@1: {count_retrieved_correct/total_count}")



