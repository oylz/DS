#!/bin/bash


ps aux | grep DS | grep -v grep | awk '{print $2}'  | xargs -i -t kill -9 {}

