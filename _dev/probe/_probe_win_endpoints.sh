#!/bin/bash
gw=$(ip route | awk '/default/ {print $3; exit}')
echo "gateway=$gw"
for p in 9393 9402; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$gw:$p/v1/models")
  echo "$p from_wsl=$code"
done
