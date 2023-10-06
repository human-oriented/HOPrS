(define (batch-resize pattern
                      new-width
                      new-height
                      offx
                      offy)
(let* ((filelist (cadr (file-glob pattern 1))))
 (while (not (null? filelist))
        (let* ((filename (car filelist))
        (image (car (gimp-file-load RUN-NONINTERACTIVE
                                    filename filename)))
        (drawable (car (gimp-image-get-active-layer image))))
        (gimp-layer-resize drawable new-width new-height offx offy)
        (gimp-image-resize-to-layers image)
        (gimp-file-save RUN-NONINTERACTIVE
                        image drawable filename filename)
        (gimp-image-delete image))
        (set! filelist (cdr filelist)))))
