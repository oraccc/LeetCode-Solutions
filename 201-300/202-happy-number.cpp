bool isHappy(int n) {
    int slow = n, fast = cal(n);
    while (fast != 1 && slow != 1 && fast != slow) {
        slow = cal(slow);
        fast = cal(cal(fast));
    }

    return (fast == 1 || slow == 1);
}

int cal(int n) {
    int tmp = 0;
    while (n >= 1) {
        tmp += (n % 10) * (n % 10);
        n = n / 10;
    }
    return tmp;
}