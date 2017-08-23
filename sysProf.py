# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:36:19 2017

@author: taxuefeini
"""

import time
import sys
from functools import wraps
  
def fn_timer(function):
    '''
    衡量函数执行时间
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        output = sys.stdout
        outputFile = open("output_time.txt", 'a')
        sys.stdout = outputFile
        print t1-t0
        outputFile.close()
        sys.stdout = output
        return result
    return function_timer
