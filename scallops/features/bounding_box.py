import pandas as pd
import shapely


def _box_to_exterior_lines(box):
    coords = box.exterior.coords
    lines = [
        shapely.LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)
    ]
    return shapely.multilinestrings(lines)


def _crosses_boundary(query, template):
    # if intersection bounds is smaller than query bounds then template is outside query
    intersection_bounds = shapely.intersection(query, template).bounds
    query_bounds = query.bounds
    query_width = query_bounds[2] - query_bounds[0]
    query_height = query_bounds[3] - query_bounds[1]
    intersection_width = intersection_bounds[2] - intersection_bounds[0]
    intersection_height = intersection_bounds[3] - intersection_bounds[1]

    return intersection_width < query_width or intersection_height < query_height


def bounding_box_to_edge_distance(
    objects_boxes_df: pd.DataFrame, objects_edges_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute distance from bounding box to nearest bounding box edge.

    :param pd.DataFrame objects_boxes_df: Dataframe with labels as index,
        and ordered columns containing AreaShape_BoundingBoxMinimum_Y,
        AreaShape_BoundingBoxMinimum_X, AreaShape_BoundingBoxMaximum_Y,
        AreaShape_BoundingBoxMaximum_X.
    :param pd.DataFrame objects_edges_df: Dataframe with labels as
        index, and ordered columns containing AreaShape_BoundingBoxMinimum_Y,
        AreaShape_BoundingBoxMinimum_X, AreaShape_BoundingBoxMaximum_Y,
        AreaShape_BoundingBoxMaximum_X.
    :return: pd.DataFrame with columns `closest_label`, `distance`, and `crosses_boundary`.
        Index values are `objects_boxes_df` labels.
    """
    # Extract coordinates from template edges
    template_coords = objects_edges_df.values
    template_labels = objects_edges_df.index.values

    template_boxes = []
    template_lines = []
    for y1, y2, x1, x2 in template_coords:
        box = shapely.box(x1, y1, x2, y2)
        template_boxes.append(box)
        template_lines.append(_box_to_exterior_lines(box))

    tree = shapely.STRtree(template_lines)

    query_coords = objects_boxes_df.values
    query_labels = objects_boxes_df.index.values

    results = []
    for i, (y1, y2, x1, x2) in enumerate(query_coords):
        query_box = shapely.box(x1, y1, x2, y2)
        indices, distances = tree.query_nearest(
            query_box,
            max_distance=None,
            return_distance=True,
            exclusive=False,
            all_matches=True,
        )
        closest_idx = indices[0]
        closest_distance = distances[0]
        crosses_boundary = False
        if closest_distance == 0:
            for idx in indices:
                if _crosses_boundary(query_box, template_boxes[idx]):
                    crosses_boundary = True
                    closest_idx = idx
                    break

        results.append(
            [
                query_labels[i],
                template_labels[closest_idx],
                closest_distance,
                crosses_boundary,
            ]
        )

    return pd.DataFrame(
        results,
        columns=["label", "closest_label", "distance", "crosses_boundary"],
    ).set_index("label")
