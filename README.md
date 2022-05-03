# gnucash
gnucash python scripts for reporting and gnucash file processing

# dscription
 Processes .gnucash file and returns a .csv file containing amount, gain, yield and their averages per period and in total for a given account and its children.
Example:
       python3 ./main.py --gnucash_file /my/path/to/file.gnucash --year 1996 --month 1 --period_type yearly --reduction_depth 1 --currency GBP --show_hidden --account_path "Big Bank"
