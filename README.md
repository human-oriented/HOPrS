# HOPrS
Human Orientated Proof Standard (HOPrS).  A collaboratively developed open standard for proving the originality and modifications that have been done to photographic and video media.  

This repository is a working one for the development of the standard containing tools, schemas, documentation and examples.  

Examples folder was an early (now abandoned) approach that looked at a series of records detailing what had been done. Each change was mapped to a previous one and hte concept at that time was that we'd have a limited number of allowable transforms that could map to a GIMP plugin.  These plugins would be computationally reversable. This would of course limit the editor to only being able to perform approved actions that were sanctioned by HOP(r)S and had the plugin provided.  This was felt to be overly restrictive.  

The POC folder is an exploration of the MQP (Merkle, QuadTree & Perceptual hash) idea as was presented at the HOPrS workshop March 2024, Cambridge University.  This doesn't Merkelise at this time but does process and match images in terms of quad tree segments.  Recommendations for getting started would be to use a PNG original image, perform modification to it.  Then use the encode_all.sh to crate suitable quad trees and then compare these with iterative_comparison.  The odd case out is cropping and that will require that you map the end image into the original, command lina parameters are given for that.  Algorithmic efficiency, code quality and efficient storage have not been considerations at this time and there are straightforward orders of magnitude improvements to easily make (your contributions welcomed!).  Suggested values to get you started are a quad tree of depth 8 for a 4000x3000 (iphone camera) image and a hamming distance of 5.  Clearly use these as you will.   

Enjoy - let us know how you get on.  
