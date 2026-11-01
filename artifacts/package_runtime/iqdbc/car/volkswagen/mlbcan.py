from iqdbc.car.volkswagen.mqbcan import (volkswagen_mqb_meb_checksum, xor_checksum,
                                         acc_hud_status_value as mqb_acc_hud_status_value,
                                         create_lka_hud_control as mqb_create_lka_hud_control)

def create_hca_steering_control(packer, bus, apply_steer, HCA_Status):
  values = {
    "HCA_01_Status_HCA": HCA_Status,
    "HCA_01_LM_Offset": abs(apply_steer),
    "HCA_01_LM_OffSign": 1 if apply_steer < 0 else 0,
    "HCA_01_Vib_Freq": 18,
    "HCA_01_Sendestatus": 1 if HCA_Status in (5, 7) else 0,
  }
  return packer.make_can_msg("HCA_01", bus, values)


def create_lka_hud_control(packer, bus, ldw_stock_values, enabled, steering_pressed, hud_alert, hud_control,
                           entering=False, special_mode=False, special_active=False):
  return mqb_create_lka_hud_control(packer, bus, ldw_stock_values, enabled, steering_pressed, hud_alert, hud_control,
                                    entering, special_mode, special_active)


def create_acc_buttons_control(packer, bus, gra_stock_values, cancel=False, resume=False, set_button=False):
  values = {s: gra_stock_values[s] for s in [
    "LS_Hauptschalter",
    "LS_Typ_Hauptschalter",
    "LS_Codierung",
    "LS_Tip_Stufe_2",
  ]}

  values.update({
    "COUNTER": (gra_stock_values["COUNTER"] + 1) % 16,
    "LS_Abbrechen": cancel,
    "LS_Tip_Wiederaufnahme": resume,
  })

  return packer.make_can_msg("LS_01", bus, values)


def acc_control_value(main_switch_on, long_active, cruiseOverride, accFaulted):
  # ACC_01.ACC_Status_ACC shares the MQB ACC_06 enum, but a fault outranks active regulation on MLB
  if cruiseOverride:
    acc_control = 4
  elif accFaulted:
    acc_control = 6
  elif long_active:
    acc_control = 3
  elif main_switch_on:
    acc_control = 2
  else:
    acc_control = 0

  return acc_control


def acc_hud_status_value(main_switch_on, acc_faulted, longActive, longOverride):
  return mqb_acc_hud_status_value(main_switch_on, acc_faulted, longActive, longOverride)


def create_acc_accel_control(packer, bus, accel, acc_control, stopping):
  acc_enabled = acc_control in (3, 4)

  acc_01_values = {
    "ACC_Status_ACC": acc_control,
    "ACC_Sollbeschleunigung": accel if acc_enabled else 0,
    "ACC_zul_Regelabw_unten": 0.2 if acc_enabled else 0,
    "ACC_zul_Regelabw_oben": 0.2 if acc_enabled else 0,
    "ACC_neg_Sollbeschl_Grad": 4.0 if acc_enabled else 0,
    "ACC_pos_Sollbeschl_Grad": 4.0 if acc_enabled else 0,
    "ACC_Dynamik": 3,
    "ACC_Anhalten": stopping if acc_enabled else False,
    "ACC_Minimale_Bremsung": 0,
  }

  return [packer.make_can_msg("ACC_01", bus, acc_01_values)]


def create_acc_hud_control(packer, bus, acc_hud_status, set_speed, leadDistance, distanceBars, fcw_alert, leadVisible,
                           unavailable, decel, d_unresponsive, hud_text=0, desired_distance=8.0, *, priority_boost=False):
  engaged = acc_hud_status in (3, 4)
  priodisp = 0 if (fcw_alert or acc_hud_status in (6, 7)) else 1 if (priority_boost or decel or d_unresponsive) else 2 if engaged else 3

  # The cluster renders the lead as a position on a fixed scale rather than a raw distance, with the
  # set follow gap sitting at mid scale. The scale spans out to 1.5x the gap before it saturates.
  if not engaged:
    acc_distance_index = 1022
  elif not leadVisible:
    acc_distance_index = 1023
  else:
    distance_ratio = leadDistance / max(desired_distance, 1.0)
    acc_distance_index = int(max(1, min(1021, round(490 * (3 - 2 * distance_ratio)))))

  values = {
    "ACC_Status_Anzeige": acc_hud_status,  # 0 off, 1 init, 2 standby, 3 active, 4 overridden, 5 shutdown reaction, 6/7 fault
    "ACC_Wunschgeschw_02": set_speed if set_speed < 250 else 327.36,  # 327.36 (raw 1023) = "no display"
    "ACC_Gesetzte_Zeitluecke": distanceBars,  # 1 aggressive, 2 standard, 3 relaxed
    "ACC_Anzeige_Zeitluecke": 1 if engaged else 0,
    "ACC_Tachokranz": 1 if engaged else 0,
    "ACC_Display_Prio": priodisp,  # 0 highest prio, 1 medium, 2 low, 3 none
    "ACC_Abstandsindex": acc_distance_index,  # 1-1020 lead distance, 1021 emergency brake, 1022 ACC off, 1023 ACC on without lead
    "ACC_Relevantes_Objekt": 2 if fcw_alert else (1 if leadVisible else 0),  # lead car: 1 green, 2 red, 0 off
    "ACC_Status_Prim_Anz": 2 if fcw_alert else (1 if engaged else 0),        # ACC symbol: 1 green, 2 red, 3 yellow, 0 off
    "ACC_Optischer_Fahrerhinweis": 1 if fcw_alert else 0,
    "ACC_Akustik": 1 if (fcw_alert or d_unresponsive) else 0,  # 0 none, 1 high prio, 2 low prio, 3 high prio continuous
    "ACC_Texte_Primaeranz": hud_text,
  }

  return packer.make_can_msg("ACC_02", bus, values)


def volkswagen_mlb_checksum(address: int, sig, d: bytearray) -> int:
  xor_starting_value = {
    0x109: 0x08, # ACC_01
    0x111: 0x10, # TSK_05
    0x30C: 0x0F, # ACC_02
    0x324: 0x27, # ACC_04
    0x10B: 0xA,  # LS_01
    0x10D: 0x0C, # ACC_05
    0x10F: 0x0E, # ACC_0x10F
    0x311: 0x12, # ACC_0x311
    0x397: 0x94, # LDW_02
    0x10C: 0x0D, # TSK_02
  }
  if address in xor_starting_value:
    return xor_checksum(address, sig, d, xor_starting_value[address])
  else:
    return volkswagen_mqb_meb_checksum(address, sig, d)
