import xlrd
wb2 = xlrd.open_workbook('jokes.xls')
sheet1 = wb2.sheet_by_name('Sheet 1')
rown=int(sheet1.nrows)
commandnames=sheet1.row_values(0)
commandnumsa=int(sheet1.ncols)
array=sheet1.col_values(0)
a=int(input())
print(rown)
