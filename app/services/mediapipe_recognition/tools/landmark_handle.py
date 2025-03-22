import copy
import itertools

def landmark_handle(landmark_list):
    """
    Calculates the offset between keypoints on the hand and normalizes the distances.

    Input:
    - landmark_list (list): A list of 21 keypoints, each with x and y coordinates, representing the hand landmarks.

    Returns:
    - list: A normalized list of the distances (offsets) between keypoints.
    """

    # Create a deep copy of the landmark list to prevent modifying the original data
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Define the distance between each pair of connected keypoints
    # Each pair [a, b] represents a connection between keypoints a and b
    distance = [[1, 0], [2, 1], [3, 1], [4, 1], [5, 0], [6, 5], [7, 5], [8, 5], [9, 0], [10, 9], [11, 9], [12, 9], [13, 0],
                [14, 13], [15, 13], [16, 13], [17, 0], [18, 17], [19, 17], [20, 17], [4, 12], [8, 12], [16, 20], [20, 12]]

    # Calculate the x and y offset for each pair of connected keypoints
    for index, coordinates in enumerate(distance):
        pos1 = temp_landmark_list[coordinates[0]]  # Get coordinates of keypoint a
        pos2 = temp_landmark_list[coordinates[1]]  # Get coordinates of keypoint b
        pos1_x, pos1_y = pos1[0], pos1[1]  # x, y coordinates of keypoint a
        pos2_x, pos2_y = pos2[0], pos2[1]  # x, y coordinates of keypoint b

        # Calculate the offset between the two keypoints
        x_offset = pos1_x - pos2_x
        y_offset = pos1_y - pos2_y
        distance[index][0] = x_offset  # Store x offset
        distance[index][1] = y_offset  # Store y offset

    # Flatten the list of distances (from 2D list to 1D list)
    distance = list(itertools.chain.from_iterable(distance))

    # Normalize the distances by dividing each value by the maximum absolute distance
    max_value = max(list(map(abs, distance)))  # Find the maximum absolute value in the list

    def normalize_(n):
        """
        Normalize the given number by dividing it by the max value.
        """
        return n / max_value

    # Apply normalization to all distance values
    distance = list(map(normalize_, distance))

    return distance
