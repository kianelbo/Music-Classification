#!/bin/bash

# sox Frantic.mp3 F.wav remix 1,2 trim 0 30

cd $1
root_dir=($(ls))
for genre in ${root_dir[*]}; do
	cd $genre
	au_files=($(ls))
	for filename in ${au_files[*]}; do
		sox $filename "${filename:0:-3}.wav"
		rm $filename
	done
	echo $genre completed
	cd ..
done
