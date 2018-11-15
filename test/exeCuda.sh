#!/bin/bash

printf "\n\n\nRunning cuda_bind\n\n\n"
./cuda_bind &&
printf "\n\n\nRunning cuda_delta\n\n\n"
./cuda_delta &&
printf "\n\n\nRunning cuda_determinant\n\n\n"
./cuda_determinant &&
printf "\n\n\nRunning cuda_epsilon\n\n\n"
./cuda_epsilon &&
printf "\n\n\nRunning cuda_identity\n\n\n"
./cuda_identity &&
printf "\n\n\nRunning cuda_index\n\n\n"
./cuda_index &&
printf "\n\n\nRunning cuda_index_map\n\n\n"
./cuda_index_map &&
printf "\n\n\nRunning cuda_init\n\n\n"
./cuda_init &&  
printf "\n\n\nRunning cuda_operators\n\n\n"
./cuda_operators &&
printf "\n\n\nRunning cuda_scalar\n\n\n"
./cuda_scalar &&
printf "\n\n\nRunning cuda_tensor\n\n\n"
./cuda_tensor &&
printf "\n\n\nRunning cuda_transpose\n\n\n"
./cuda_transpose &&
printf "\n\n\nRunning cuda_trees\n\n\n"
./cuda_trees 
