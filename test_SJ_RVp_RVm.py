import basicFunction as bf

all_data = bf.getdata('0910')
all_data = bf.concatRV(all_data)

RVp, RV = bf.calculRV(all_data, type='RV+')
RVm, _ = bf.calculRV(all_data, type='RV-')
SJ, _ = bf.calculRV(all_data, type='SJ')

