#!/bin/bash

cd $1
root_dir=($(ls))
for genre in ${root_dir[*]}; do
	cd $genre
	au_files=($(ls))
	index=0

	for filename in ${au_files[*]}; do
		# 1st sample from third 30 seconds of the song
		printf -v j "%04d" $index
		sox $filename "${genre}_${j}.mp3" remix 1,2 trim 60 30
		((index++))
		# 2nd sample from second last 30 seconds of the song
		printf -v j "%04d" $index
		sox $filename "${genre}_${j}.mp3" remix 1,2 trim -60 30
		((index++))
		# removing the original file
		rm $filename
	done

	echo $genre completed
	cd ..
done
