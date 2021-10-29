import numpy as np
def Distance(bbox1, bbox2):
    xmin = np.min([bbox1[0],bbox2[0]])
    ymin = np.min([bbox1[1],bbox2[1]])
    xmax = np.max([bbox1[2],bbox2[2]])
    ymax = np.max([bbox1[3],bbox2[3]])
    return np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)

def Center_Distance(bbox1, bbox2):
    center1 = (bbox1[:2]+bbox1[2:])/2
    center2 = (bbox2[:2]+bbox2[2:])/2
    return np.sqrt(np.sum((center1-center2)**2))

def get_relation_point(bbox1, bbox2, center):
    if center:
        center1 = (bbox1[:2]+bbox1[2:])/2
        center2 = (bbox2[:2]+bbox2[2:])/2
        xmin = np.min([center1[0],center2[0]])
        ymin = np.min([center1[1],center2[1]])
        xmax = np.max([center1[0],center2[0]])
        ymax = np.max([center1[1],center2[1]])
        return [xmin, ymin, xmax-xmin, ymax-ymin]
    else:
        xmin = np.min([bbox1[0],bbox2[0]])
        ymin = np.min([bbox1[1],bbox2[1]])
        xmax = np.max([bbox1[2],bbox2[2]])
        ymax = np.max([bbox1[3],bbox2[3]])
        return [xmin, ymin, xmax-xmin, ymax-ymin]
    
# def get_relation_points(bboxes,center):
#     edges = []
#     added_points = [0]
#     rest_points = list(np.arange(1,len(bboxes)))
#     while len(rest_points)>0:
#         rest_point = rest_points[0]
#         added_point = added_points[-1]
#         points = [added_point, rest_point]
#         added_points.append(points[1])
#         rest_points.remove(points[1])
#         edges.append(points)
#     relation_points = [get_relation_point(bboxes[edge[0]], bboxes[edge[1]], center) for edge in edges]
#     return relation_points


def get_relation_points(bboxes,center):
    edges = []
    added_points = [0]
    rest_points = list(np.arange(1,len(bboxes)))
    while len(rest_points)>0:
        mini = 10000
        points = []
        for added_point in added_points:
            for rest_point in rest_points:
                d = Center_Distance(bboxes[rest_point], bboxes[added_point])
#                 if center:
#                     d = Center_Distance(bboxes[rest_point], bboxes[added_point])
#                 else:
#                     d = Distance(bboxes[rest_point], bboxes[added_point])
                if d<mini:
                    mini = d
                    points = [added_point, rest_point]
        added_points.append(points[1])
        rest_points.remove(points[1])
        edges.append(points)
    relation_points = [get_relation_point(bboxes[edge[0]], bboxes[edge[1]], center) for edge in edges]
    return relation_points
