#!/bin/zsh

#Encode all the files in the current directory matching *.png 
#Will create a metric tonne of tmp....jpeg files that are all the segments. 
#Example file is
#tmp.D2.1-.pdq.4872574c95b1ab663bc954bf8d50a94dceb3d2cc4d22a2d92494936b5b6d49b6.hawker_fury.jpeg.segment.0.0.2016.1512.png
#they all start tmp. which is useful for purging them
#the D2 suggests depth 2 in the quadtree
#the 1- is the part of the quadtree path.  May also be 1-4-3-1-2-3-1- etc.  
#pdq is the hashing algorithm used
#4872574c95b1ab663bc954bf8d50a94dceb3d2cc4d22a2d92494936b5b6d49b6 is the perceptual hash
#hawker_fury.png is the basename of the file used.  If you've specified ../../somepath/image.png it'll just be image.png
#segment.0.0.2016.1512 just describes the segment and from x0,y0,x1,y1 (this happens to be the top left quarter of an image)

#This previously used to contain a cryptographic hash - this is no longer present, ordering will happen based on the 
#depth and path parts. 
#the above is just debug to verify visually the comparisons.

#Will create a file that is <filename>.hoprs which has the quadtree in 

echo "Starting to process pngs"
for file in *.png;do python ../../encode_file_to_depth.py $file 10 4032 3024 && echo $file ; done 

echo "Starting to process jpgs"
for file in *.jpg;do python ../../encode_file_to_depth.py $file 10 4032 3024 && echo $file ; done 

echo "Starting to process jpegs"

for file in *.jpeg;do python ../../encode_file_to_depth.py $file 10 4032 3024 && echo $file ; done 

echo Completed you now have the following .hoprs files
ls -l *.hoprs

#From the output of this you can inspect each segment in different resolutions of the same file to be confident that 
#the right segments have ben generated and that the perceptual hashes are reasonable

