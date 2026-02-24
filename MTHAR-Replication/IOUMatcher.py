import numpy as np

class IOUMatcher:
    """
    Implements greedy one-to-one assignment between
    generated windows and ground-truth bounding boxes
    using IOU matrix.
    """

    def __init__(self, windows, truths, threshold=0.5):
        self.windows = windows
        self.truths = truths
        self.threshold = threshold

        self.na = len(windows)
        self.nb = len(truths)

        self.M = None
        self.phase1_assignments = {}
        self.phase2_assignments = {}

    # ---------------------------------
    # IOU Computation
    # ---------------------------------
    @staticmethod
    def interval_iou(window, truth):
        w_start, w_end = window
        t_start, t_end = truth

        inter_start = max(w_start, t_start)
        inter_end = min(w_end, t_end)

        intersection = max(0, inter_end - inter_start)
        union = (w_end - w_start) + (t_end - t_start) - intersection

        if union == 0:
            return 0.0

        return intersection / union

    # ---------------------------------
    # Build IOU Matrix
    # ---------------------------------
    def build_matrix(self):
        M = np.zeros((self.na, self.nb))

        for i in range(self.na):
            for j in range(self.nb):
                M[i, j] = self.interval_iou(
                    self.windows[i], self.truths[j]
                )

        self.M = M
        return M

    # ---------------------------------
    # Phase 1: Greedy Matching
    # ---------------------------------
    def phase1_greedy_assignment(self):

        if self.M is None:
            self.build_matrix()

        M_work = self.M.copy()

        used_rows = set()
        used_cols = set()

        for _ in range(self.nb):

            masked_M = M_work.copy()

            for r in used_rows:
                masked_M[r, :] = -1
            for c in used_cols:
                masked_M[:, c] = -1

            i, j = np.unravel_index(np.argmax(masked_M), masked_M.shape)

            if masked_M[i, j] <= 0:
                break

            self.phase1_assignments[i] = j
            used_rows.add(i)
            used_cols.add(j)

    # ---------------------------------
    # Phase 2: Threshold Assignment
    # ---------------------------------
    def phase2_remaining_assignment(self):

        remaining_windows = (
            set(range(self.na)) -
            set(self.phase1_assignments.keys())
        )

        for i in remaining_windows:
            j = np.argmax(self.M[i])
            max_iou = self.M[i, j]

            if max_iou >= self.threshold:
                self.phase2_assignments[i] = j

    # ---------------------------------
    # Run Full Matching
    # ---------------------------------
    def match(self):

        self.build_matrix()
        self.phase1_greedy_assignment()
        self.phase2_remaining_assignment()

        return self.phase1_assignments, self.phase2_assignments