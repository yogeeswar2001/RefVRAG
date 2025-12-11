import networkx as nx

def calculate_xmax_ymax(bbox):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return x_max, y_max

def calculate_spatial_relations(bbox_a, bbox_b):
    x_min_a, y_min_a, width_a, height_a = bbox_a
    x_min_b, y_min_b, width_b, height_b = bbox_b
    
    x_max_a, y_max_a = calculate_xmax_ymax(bbox_a)
    x_max_b, y_max_b = calculate_xmax_ymax(bbox_b)
    
    spatial_relations = []
    
    if x_min_a < x_max_b and x_max_a > x_min_b and y_min_a < y_max_b and y_max_a > y_min_b:
        spatial_relations.append("overlaps")

    if x_max_a < x_min_b:
        spatial_relations.append("left_of")

    if x_min_a > x_max_b:
        spatial_relations.append("right_of")

    if y_max_a < y_min_b:
        spatial_relations.append("above")

    if y_min_a > y_max_b:
        spatial_relations.append("below")
    
    return spatial_relations


def relation_to_text(source_id, source_label, relation_type, target_id, target_label):
    if relation_type == "overlaps":
        return f"Object {source_id} ({source_label}) overlaps with Object {target_id} ({target_label})."
    elif relation_type == "left_of":
        return f"Object {source_id} ({source_label}) is to the left of Object {target_id} ({target_label})."
    elif relation_type == "right_of":
        return f"Object {source_id} ({source_label}) is to the right of Object {target_id} ({target_label})."
    elif relation_type == "above":
        return f"Object {source_id} ({source_label}) is above Object {target_id} ({target_label})."
    elif relation_type == "below":
        return f"Object {source_id} ({source_label}) is below Object {target_id} ({target_label})."
    elif relation_type == "same_object_type":
        return f"Object {source_id} ({source_label}) is of the same type as Object {target_id} ({target_label})."
    else:
        return f"Object {source_id} ({source_label}) is related to Object {target_id} ({target_label})."

def generate_scene_graph_description(objects, include_location=True, include_relations=True, include_count=True):

    graph = nx.DiGraph()
    object_counter = {}

    for obj in objects:
        graph.add_node(obj['id'], label=obj['label'], bbox=obj['bbox'])

        label = obj['label']
        if label in object_counter:
            object_counter[label] += 1
        else:
            object_counter[label] = 1

    for node_a, data_a in graph.nodes(data=True):
        for node_b, data_b in graph.nodes(data=True):
            if node_a < node_b:
                bbox_a = data_a['bbox']
                bbox_b = data_b['bbox']
                relations = calculate_spatial_relations(bbox_a, bbox_b)
                
                for relation in relations:
                    graph.add_edge(node_a, node_b, relation=relation)

    descriptions = []

    if include_location:
        for node, data in graph.nodes(data=True):
            label = data.get('label', 'unknown object')
            bbox = data.get('bbox', [])
            description = f"Object {node} is a {label} located at coordinates [{bbox[0]}, {bbox[1]}] with dimensions {bbox[2]}x{bbox[3]}."
            descriptions.append(description)
    
    if include_relations:
        for source, target, data in graph.edges(data=True):
            relation_type = data.get('relation', 'related to')
            source_label = graph.nodes[source]['label']
            target_label = graph.nodes[target]['label']
            description = relation_to_text(source, source_label, relation_type, target, target_label)
            descriptions.append(description)

    if include_count:
        count_description = "Object counting:\n"
        for label, count in object_counter.items():
            count_description += f"- {label}: {count}\n"
        descriptions.append(count_description)
    
    return "\n".join(descriptions)
