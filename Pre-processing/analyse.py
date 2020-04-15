from coco_analyse import cocoanalyse
import argparser

args = argparser.parse_cmdline_args('analyse')

analyse = cocoanalyse(args.ann_path)
analyse.getinfo()
analyse.displayanndist(x_lab=args.xa_lab,y_lab=args.ya_lab,title=args.a_title,output_path=args.out_path)
analyse.displaycatdist(x_lab=args.xc_lab,y_lab=args.yc_lab,title=args.c_title,output_path=args.out_path)
