#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 \"<command>\" [\"<name>\"] [\"<comment>\"]"
  exit 1
fi

DIR="./output"
DATE=$(date +"%Y%m%d")
COMMAND="$1"
NAME="${2:-etc}"
COMMENT="$3"


LOG_DIR="${DIR}/nohup/$NAME"
mkdir -p $LOG_DIR
LOG="${LOG_DIR}/${DATE}.log"
OUT="${LOG_DIR}/${DATE}.out"
ERR="${LOG_DIR}/${DATE}.err"

RUN_TIME=$(date +"%Y/%m/%d %H:%M:%S")

SEPARATOR=$(printf '#%.0s' {1..40})

{
    echo -e "\n\n$SEPARATOR$SEPARATOR$SEPARATOR\n$SEPARATOR$SEPARATOR$SEPARATOR\n$SEPARATOR$SEPARATOR$SEPARATOR\n$SEPARATOR"
    echo -e "# Run Time: $RUN_TIME"
    echo -e "# Command: $COMMAND"
    if [ -n "$COMMENT" ]; then
        echo -e "# Comment: $COMMENT"
    fi
    echo "$SEPARATOR"
} | tee -a "$LOG" "$OUT" "$ERR" > /dev/null


# nohup bash -c "$COMMAND" 1> >(tee -a "$LOG" "$OUT" > /dev/null) 2> >(tee -a "$LOG" "$ERR" > /dev/null) &
nohup bash -c "$COMMAND" 1> >(tee -a "$LOG" "$OUT") 2> >(tee -a "$LOG" "$ERR") &
echo "\"$COMMAND\" to \"$NAME\" started with PID \"$!\""


# Example usage:
# ./nohup.sh "{command}" "{name}" "{comment}"