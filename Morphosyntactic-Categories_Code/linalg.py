import numpy as np
import numpy.linalg as npl


def manhattan_normalizer(vec):
    c = sum(vec)
    if c == 0:
        return vec
    return list(map(lambda n: n / c, vec))


def project(vector, vector_space):
    return vector_space.dot(npl.inv(vector_space.T.dot(vector_space))).dot(vector_space.T).dot(vector)


def is_in_cone(vector, vector_space):
    return np.count_nonzero(vector_space.dot(project(vector, vector_space)) < 0) == 0


def angle(vector, vector_space):
    projection = project(vector, vector_space)
    norms = (npl.norm(vector) * npl.norm(projection))
    return np.dot(vector, projection) / norms if norms else np.nan


def dict_distance(v1, v2):
    v = {}
    for k in v1:
        v[k] = v.get(k, 0.) + v1[k]
    for k in v2:
        v[k] = v.get(k, 0.) - v2[k]
    return npl.norm(np.array([v[t] for t in v]))


def distance(v1, v2):
    return npl.norm(v1 - v2)
