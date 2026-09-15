#!/usr/bin/env bash
set -euo pipefail
exec python3 /home/a/ct/va/run_w2/servo_rtc_deploy_kalman_test2_vel.py \
  --deploy /home/a/ct/va/outputs/fm_openarm_hcx_dual_arm_with_out_room_s_depth_nobs1_h32_nact16_statedelta_260912170848/servo_rtc_deploy_kalman.yaml \
  "$@"
