#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2020/3/14-20:36
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : wechat：csy_lgy  CSDN：https://me.csdn.net/sinat_26745777 
# @Project  : Bible-时间复杂度和空间复杂度
# endregion
int a =0, b = 0
for(i=0; i < N; i++){
    a = a + radn()
    b = b + rand()
}


for(j = 0; j < N/2; j++){
    b = b + rand()
}

int a = 0
for(i = 0; i , N; i++){
    for(j = N; j > i; j--){
        a = a + i + j;
    }
}

//时间复杂度：O(N^2)
//空间复杂度：O（1）


//---------------------------------
int a = 0, i = N;
while(i > 0){
    a += i; // 1个操作
    i /= 2; // 1个操作
}

N = 40; i = 40;
//
//i=20 2
//i=10 2
//i=5 2
//i=2 2
//i=1 2
//i=0 2
//
//2*6===2*log(n)
//O(nlog(n))

X:O(log n) > Y: O(n)
X:O(n log n) > Y:o(n^2)


//定理：if X的时间复杂度要由于y的时间复杂度，那么，假设存在一个足够大的数M，
//当n>M时，我么可以保证X的实际效率要由于Y的实际效率


C * O(N) if only if C跟N没有相关性
O(1)
O(log n)  tree heap binary serach
O(n)
O(n log n)    quicksort  heapost
O(n^2)
O(n^3)
O(2^n)
O(3^n)
