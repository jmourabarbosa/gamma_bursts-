cd /home/user/pm-off
export PM_VARIANT=os
( python -m paper_model.natural_readout ; python -m paper_model.off_vs_memory ) > /home/user/fits_os/b1.log 2>&1 &
( python -m paper_model.single_dip ; python -m paper_model.single_dip_readout ; python -m paper_model.fig_unit_sets_variant ; python -m paper_model.dips_vs_states ) > /home/user/fits_os/b2.log 2>&1 &
python -m paper_model.tau_sweep 20 30 40 50 60 80 100 120 > /home/user/fits_os/b3.log 2>&1 &
python -m paper_model.tau_sweep 150 200 250 300 400 500 750 1000 1500 > /home/user/fits_os/b4.log 2>&1 &
wait
python -m paper_model.fit_tau_pooled > /home/user/fits_os/b5.log 2>&1
echo BATCH DONE
