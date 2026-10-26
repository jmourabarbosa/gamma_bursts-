cd /home/user/pm-off
while pgrep -f "[p]ython -m paper_model.fit_session" >/dev/null; do sleep 60; done
S=($(ls /home/user/fits_session/*.json | xargs -n1 basename | sed 's/.json//'))
for w in 0 1 2 3; do L=""; for i in "${!S[@]}"; do [ $((i % 4)) -eq $w ] && L="$L ${S[$i]}"; done
  python -m paper_model.session_tau sweep /home/user/fits_session $L > /home/user/fits_session/sweep$w.log 2>&1 & done
wait
python -m paper_model.session_tau fit /home/user/fits_session 2>&1 | grep -v -i "warn\|findfont"
