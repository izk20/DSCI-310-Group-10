# author: Ahmed Rizk
# date: 2022-03-31
# https://onedrive.live.com/download?cid=3186CCDB0C6495E0&resid=3186CCDB0C6495E0%2157273&authkey=AK4_vAlM4AFx7_M


"Downloads csv data from the web to a local filepath as a csv format

Usage: src/download_dataset_script.R --url=<url> --out_dir=<out_dir>

Options:
--url=<url>           URL to where the dataset is hosted
--out_dir=<out_dir>   Path to directory where the data should be written
" -> doc

library(docopt)
opt <- docopt(doc)


download.file(opt$url, opt$out_dir)