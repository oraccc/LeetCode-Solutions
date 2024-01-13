int rand10() {
    int n = (rand7()-1) * 7 + rand7();
    if (n>40) return rand10();
    return (n-1)/4+1;
}