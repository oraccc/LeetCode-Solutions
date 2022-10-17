bool judgeSquareSum(int c) {
    long min = 0, max = sqrt(c);
    while (min != max) {
        if (min * min + max * max == c) return true;
        else if (min * min + max * max < c) ++min;
        else --max; 
    }
    if (min * min + max * max == c) return true;
    else return false;
}