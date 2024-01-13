int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
    if (ax2 <= bx1 || by2 <= ay1 || bx2 <= ax1 || ay2 <= by1) {
        return (ax2-ax1) * (ay2-ay1) + (bx2-bx1) * (by2-by1);
    }
    int xOverlap = min(ax2-ax1, min(ax2-bx1, min(bx2-ax1, bx2-bx1)));
    int yOverlap = min(ay2-ay1, min(by2-ay1, min(ay2-by1, by2-by1)));
    int areaOverlap = xOverlap * yOverlap;

    return (ax2-ax1) * (ay2-ay1) + (bx2-bx1) * (by2-by1) - areaOverlap;
}