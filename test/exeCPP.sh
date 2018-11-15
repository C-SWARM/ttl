#!/bin/bash

printf "\n\nRunning bind\n\n\n"
./bind &&
printf "\n\nRunning delta\n\n\n"
./delta &&
printf "\n\nRunning epsilon\n\n\n"
./epsilon &&
printf "\n\nRunning index\n\n\n"
./index &&
printf "\n\nRunning index_map\n\n\n"
./index_map &&
printf "\n\nRunning identity\n\n\n"
./identity &&
printf "\n\nRunning library\n\n\n"
./library &&
printf "\n\nRunning operators\n\n\n"
./operators &&
printf "\n\nRunning scalar\n\n\n"
./scalar &&
printf "\n\nRunning tensor\n\n\n"
./tensor &&
printf "\n\nRunning trees\n\n\n"
./trees

