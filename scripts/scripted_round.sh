#!/bin/bash
# One round of scripted data between collector cycles (rl_loop --between-cmd): the next GENERATION of full-reward
# episodes of every scripted policy (rollout.py --scripted --store-success-only), as chunks
# runs/<run>/scripted/g<GGGG>_<part>_<policy>/ (READY when complete; bucket_daemon.py --role collector pushes them to
# bucket scripted/<run>/ and drops the local stores). A round completes one generation (an interrupted one is finished
# first; finished chunks are skipped). A chunk starts only with MIN_FREE_GB free and fewer than MAX_UNPUSHED chunks
# waiting for the upload; after MAX_WAIT_MIN minutes of waiting the round gives up (the collect goes ahead).
#
# A generation's spec is frozen in scripted/g<GGGG>.plan when it starts (edits of the plan file apply from the next
# generation). Its chunks, in making order (chunks_of):
#   plain    FIRST (generation 1) / PER episodes per policy in chunks of at most PART, parts outermost (every task's
#            first part comes first), seeds S0 + 200000 + G * 10000 + ...
#   inject   "inject <policy> N" (every generation, at most 4 chunks of PART): N programmatic hard cases (rollout.py
#            --inject, sim/inject.py: the scripted policy with one injected error, then recovery's Retry), seeds 120M +
#            task * 1M + G * 20000 + chunk * 5000, chunks g<GGGG>_9<k>_<policy>; jobs that reach no error state give no
#            episode. Default: liftpot 500, stack 500 (tasks whose failures are mostly execution errors); any inject
#            line in the plan replaces the default, "inject off" = none
#
# The trainer weighs each generation like a collect (flywheel.py) and trains on the newest WINDOW generations (keep
# WINDOW equal to rl_loop --scripted-window); when a round starts generation G, generations up to G - WINDOW go to
# bucket scripted/<run>/RETIRED.txt (scripts/scripted_data.py retire) and the trainer's bucket_daemon deletes its
# local copies.
#
#   scripts/scripted_round.sh <run> <workers> <render gpus> <cpus>
#   runs/<run>/scripted_plan.txt (optional) "key value" lines, # comments; the defaults:
#     first 2000 / per 1000 / part 1000 / window 8 / min_free_gb 40 / max_unpushed 4 / max_wait_min 60
#     policies <all 8> / inject liftpot 500 / inject stack 500
# Exit status: 0 = the generation is complete (and retirements went through), 1 = not (FAILED chunks, a wait that gave
# up, a failed retirement); the last line sums the round up (rl_loop logs it).
#
# Replace this file by rename (new inode), never edit it in place while it runs: bash reads a running script lazily.
set -uo pipefail
RUN=$1 WORKERS=$2 GPUS=$3 CPUS=$4
SOL=$(cd "$(dirname "$0")/.." && pwd)
OUT=$SOL/../runs/$RUN/scripted
mkdir -p "$OUT"
FIRST=2000 PER=1000 PART=1000 WINDOW=8 MIN_FREE_GB=40 MAX_UNPUSHED=4 MAX_WAIT_MIN=60
INJ="liftpot:500 stack:500" INJ_PLAN="" UNUSED=""
IP0=90  # chunk parts of the injected hard cases: IP0 + 1 ..
POLICIES="packbox stack liftpot lifttray rotatevalve cubehandover pickbook shelfbook"
PLAN=$SOL/../runs/$RUN/scripted_plan.txt
if [ -f "$PLAN" ]; then
  while read -r key val; do
    case $key in
      first) FIRST=$val ;; per) PER=$val ;; part) PART=$val ;; window) WINDOW=$val ;;
      min_free_gb) MIN_FREE_GB=$val ;; max_unpushed) MAX_UNPUSHED=$val ;; max_wait_min) MAX_WAIT_MIN=$val ;;
      policies) POLICIES=$val ;;
      inject) read -r xp xn <<< "$val"; INJ_PLAN="$INJ_PLAN $xp:$xn" ;;
      "") ;;
      *) UNUSED="$UNUSED $key" ;;
    esac
  done < <(grep -v '^[[:space:]]*#' "$PLAN")
fi
[ -n "$INJ_PLAN" ] && INJ=${INJ_PLAN//off:/}
log() { echo "$(date '+%F %T') $*"; }
[ -n "$UNUSED" ] && log "plan keys not used (ignored):$UNUSED"
# scripted policy $1 (sim/scripted.py) -> TASK, its seed block S0 (generations S0 + 200000 + ...) and FI, the task's
# index in data/tasks.TASK_FAMILIES; returns 1 if unknown
task_of() {
  case $1 in
    liftpot) TASK=LiftPot S0=73000000 FI=0 ;;
    lifttray) TASK=LiftTray S0=74000000 FI=1 ;;
    stack) TASK=StackTwoBlocks S0=71000000 FI=2 ;;
    shelfbook) TASK=StackSingleBookShelf S0=77000000 FI=3 ;;
    cubehandover) TASK=CubeHandover S0=75000000 FI=4 ;;
    packbox) TASK=PackBox S0=70000000 FI=5 ;;
    pickbook) TASK=PickSingleBookFromTable S0=76000000 FI=6 ;;
    rotatevalve) TASK=RotateValve S0=72000000 FI=7 ;;
    *) return 1 ;;
  esac
}

spec_file() { printf "%s/g%04d.plan" "$OUT" "$1"; }
# generation $1's spec -> GP (policies), GN (episodes per policy), GPART (chunk size), GI (injected hard cases per
# policy): its frozen g<GGGG>.plan (the current settings before it is written)
spec() {
  GP=$POLICIES GPART=$PART
  declare -gA GI=()
  if [ "$1" -eq 1 ]; then GN=$FIRST; else GN=$PER; fi
  [ -f "$(spec_file "$1")" ] || return 0
  local key val xp xn
  while read -r key val; do
    case $key in policies) GP=$val ;; episodes) GN=$val ;; part) GPART=$val ;;
      inject) read -r xp xn <<< "$val"; GI[$xp]=$xn ;;
    esac
  done < "$(spec_file "$1")"
}
chunk() { printf "%s/g%04d_%d_%s" "$OUT" "$1" "$2" "$3"; }  # generation, part, policy
# generation $1's chunks in making order, one per line: "<dir> <policy> <episodes> <seed> [rollout.py args]"
chunks_of() {
  spec "$1"
  local g=$1 p pol np ni k
  np=$(( (GN + GPART - 1) / GPART ))  # parts per policy
  for ((p = 1; p <= np; p++)); do
    for pol in $GP; do
      task_of "$pol" || continue  # seeds stay below the next policy's S0 while G < 80
      echo "$(chunk "$g" "$p" "$pol") $pol $(( p < np ? GPART : GN - (np - 1) * GPART )) \
$((S0 + 200000 + g * 10000 + (p - 1) * GPART))"
    done
  done
  for pol in $GP; do  # injected hard cases
    task_of "$pol" || continue
    ni=$(( (${GI[$pol]:-0} + GPART - 1) / GPART ))
    for ((k = 1; k <= ni; k++)); do  # seeds stay inside the task's 1M block while G < 50
      echo "$(chunk "$g" $((IP0 + k)) "$pol") $pol $(( k < ni ? GPART : ${GI[$pol]} - (ni - 1) * GPART )) \
$((120000000 + FI * 1000000 + g * 20000 + (k - 1) * 5000)) --inject 1"
    done
  done
}
# generation $1 has every chunk of its spec (READY: generated, or PUSHED: in the bucket)
has_all() {
  local d rest
  while read -r d rest; do
    [ -e "$d/READY" ] || [ -e "$d/PUSHED" ] || return 1
  done < <(chunks_of "$1")
}
unpushed() { find "$OUT" -mindepth 2 -maxdepth 2 -name READY -printf '%h\n' | while read -r d; do
  [ -e "$d/PUSHED" ] || echo "$d"; done | wc -l; }
# 0 once there is room for a chunk; 1 after MAX_WAIT_MIN minutes without (the round gives up, the collect goes ahead)
wait_disk() {
  local n=0 free
  while :; do
    free=$(df -BG --output=avail "$OUT" | tail -n 1 | tr -dc 0-9)
    [ "$free" -ge "$MIN_FREE_GB" ] && [ "$(unpushed)" -lt "$MAX_UNPUSHED" ] && return 0
    [ $((n * 30)) -ge $((MAX_WAIT_MIN * 60)) ] && return 1
    [ $((n % 20)) -eq 0 ] && log "waiting for the upload: ${free} GB free, $(unpushed) chunks not pushed yet"
    n=$((n + 1)); sleep 30
  done
}

G=$(find "$OUT" -mindepth 1 -maxdepth 1 \( -name 'g[0-9][0-9][0-9][0-9]_*' -o -name 'g[0-9][0-9][0-9][0-9].plan' \) \
  -printf '%f\n' | cut -c2-5 | sort -n | tail -n 1)
G=$((10#${G:-0}))
# this round's generation: the newest one if unfinished, else the next, whose spec is frozen now
if [ "$G" -eq 0 ] || has_all "$G"; then
  G=$((G + 1))
  spec "$G"
  KNOWN=""
  for pol in $POLICIES; do
    if task_of "$pol"; then KNOWN="$KNOWN $pol"; else log "unknown scripted policy $pol: not in generation $G"; fi
  done
  XLINES=""
  for tok in $INJ; do  # recurring: every generation, N more injected-hard-case jobs (<= 4 chunks)
    xp=${tok%%:*} xn=${tok#*:}
    if [[ " $KNOWN " == *" $xp "* ]] && [ "$xn" -gt 0 ] 2>/dev/null; then
      [ "$xn" -gt $((4 * PART)) ] && xn=$((4 * PART))
      XLINES="${XLINES}inject $xp $xn"$'\n'
    else
      log "inject $tok: not a policy of generation $G (or not a count): ignored"
    fi
  done
  printf "policies %s\nepisodes %s\npart %s\n%s" "${KNOWN# }" "$GN" "$PART" "$XLINES" > "$(spec_file "$G").tmp" \
    && mv "$(spec_file "$G").tmp" "$(spec_file "$G")"
fi
spec "$G"
STATUS=0 MADE=0 FAILED=0 NOTES=""
# with generation G on its way the trainer's window holds G-WINDOW+1..G: older generations leave training and disk
for ((old = 1; old <= G - WINDOW; old++)); do
  [ -e "$OUT/$(printf 'g%04d' "$old").retired" ] && continue
  log "retiring generation $old (window $WINDOW, generation $G next)"
  if "$SOL/.venv/bin/python" "$SOL/scripts/scripted_data.py" retire --run "$RUN" --generation "$old"; then
    touch "$OUT/$(printf 'g%04d' "$old").retired"
  else
    log "retiring generation $old FAILED (retried next round)"
    STATUS=1 NOTES="$NOTES; retiring generation $old failed"
  fi
done
XLOG=""
for pol in "${!GI[@]}"; do XLOG="$XLOG, +${GI[$pol]} injected hard cases $pol"; done
log "generation $G: $GN episodes per policy in $(( (GN + GPART - 1) / GPART )) chunk(s) of <= $GPART$XLOG; \
policies $GP"
# make chunk $1 ($2 episodes from seed $3; more rollout.py args after them) of policy $POLICY: READY when done
make_chunk() {
  local d=$1 eps=$2 seed=$3
  shift 3
  rm -rf "$d"
  log "chunk $(basename "$d"): $eps episodes from seed $seed${1:+ ($*)}"
  if ! taskset -c "$CPUS" "$SOL/.venv-sim/bin/python" "$SOL/scripts/rollout.py" --scripted "$POLICY" --tasks "$TASK" \
      --episodes-per-task "$eps" --seed-start "$seed" --workers "$WORKERS" --envs-per-worker 1 --env-cache 1 \
      --gpus "$GPUS" --record --store-success-only --out "$d" "$@"; then
    log "chunk $(basename "$d") FAILED (redone next round)"
    FAILED=$((FAILED + 1))
    return 1
  fi
  touch "$d/READY"
  MADE=$((MADE + 1))
  log "chunk $(basename "$d") READY: $(python3 -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1])]; \
print(sum('store' in x for x in e), 'full-reward episodes stored of', len(e))" "$d/episodes.jsonl")"
}
mapfile -t LIST < <(chunks_of "$G")
for line in "${LIST[@]}"; do
  read -r d POLICY eps seed args <<< "$line"
  [ -e "$d/READY" ] || [ -e "$d/PUSHED" ] && continue
  task_of "$POLICY" || { log "unknown scripted policy $POLICY"; FAILED=$((FAILED + 1)); continue; }
  if ! wait_disk; then
    log "no room for chunk $(basename "$d") after $MAX_WAIT_MIN min: the rest of the round is skipped"
    NOTES="$NOTES; gave up waiting for disk / upload"
    break
  fi
  # shellcheck disable=SC2086  # args: "--inject 1"
  make_chunk "$d" "$eps" "$seed" $args
done
if has_all "$G"; then DONE_MSG="complete"; else DONE_MSG="incomplete (continued next round)"; STATUS=1; fi
log "round summary: generation $G $DONE_MSG; $MADE chunks made, $FAILED failed$NOTES"
exit $STATUS
