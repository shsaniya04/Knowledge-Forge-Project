import numpy as np

vector_a = np.array([1, 0, 1])
vector_b = np.array([1, 1, 0])
vector_c = np.array([1, 0, 1])

similarity_ab = np.dot(vector_a, vector_b) / (
    np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
)

similarity_ac = np.dot(vector_a, vector_c) / (
    np.linalg.norm(vector_a) * np.linalg.norm(vector_c)
)

print("Similarity A-B:", similarity_ab)
print("Similarity A-C:", similarity_ac)