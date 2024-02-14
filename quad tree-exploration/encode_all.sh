#!/bin/zsh

#Encode all the files in the current directory matching *.jpeg (NOT JPG! NOR JPEG)
#Will create a metric tonne of tmp....jpeg files that are all the segments. 
#Example file is
#tmp.uniq.6.f65744dc69c6df1ccc6925b4f4abedbf8b19213ed0977ab542fe51fb65b79f48_pdq.be22aaeaf8cefbe7f377d7775d771df79c679a628a22ba22a802980290029002_hawker_fury.jpeg_tmp_segment_504.0.630.94.jpg
#they all start tmp. which is useful for purging them
#the uniq.6.f65744dc69c6df1ccc6925b4f4abedbf8b19213ed0977ab542fe51fb65b79f48_ part says that the file is depth 6 
#(which is 5 down from the quadtree root), the f65744dc69c6df1ccc6925b4f4abedbf8b19213ed0977ab542fe51fb65b79f48 is 
#a crpytographc hash of the coordinate part which is a cheat ONLY to allow these to sort appropriately in a directory so 
#all the different resolutions of a similar test image appear together for an ls -l this is not a perceptual hash  
#_pdq.be22aaeaf8cefbe7f377d7775d771df79c679a628a22ba22a802980290029002_hawker_fury.jpeg_tmp_segment_504.0.630.94.jpg
#this part says that pdq is the perceptual hash algorithm used.  The pdq hash is be22....9002
#the original file was hawker_fury.jpeg  And this is a segment that is the area of the original file that goes from
#x0,y0,  x1,y1 504,0  630,94 which is a location in pixels


#the above is just debug to verify visually the comparisons.

#Will also create a file that is <filename>.hoprs which has the quadtree in 

for file in *.jpeg;do python ../../encode_file_to_depth.py $file 5 4032 3024 | tee $file.hoprs ; done 

#From the output of this you can inspect each segment in different resolutions of the same file to be confident that 
#the right segments have ben generated and that the perceptual hashes are reasonable

