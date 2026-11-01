# Copyright (c) 2026 IQ.Lvbs. All rights reserved.
import base64
import io
import pprint
import webbrowser
import matplotlib.pyplot as plt


def report_header(title, platform, route, _description, CP, ID):
  builder = [
    "<style>summary { cursor: pointer; }\n td, th { padding: 8px; } </style>\n",
    f"<h1>{title} maneuver report</h1>\n",
    f"<h3>{platform}</h3>\n",
    f"<h3>{route}</h3>\n",
    f"<h3>{ID.gitCommit}, {ID.gitBranch}, {ID.gitRemote}</h3>\n",
  ]
  if _description is not None:
    builder.append(f"<h3>Description: {_description}</h3>\n")
  car_params = pprint.pformat({k: v for k, v in CP.to_dict().items() if not k.endswith("DEPRECATED")}, indent=2)
  builder.append(f"<details><summary><h3 style='display: inline-block;'>CarParams</h3></summary><pre>{car_params}</pre></details>\n")
  builder.append('{ summary }')
  return builder


def embed_figure(fig):
  buffer = io.BytesIO()
  fig.savefig(buffer, format='webp')
  plt.close(fig)
  buffer.seek(0)
  return f"<img src='data:image/webp;base64,{base64.b64encode(buffer.getvalue()).decode()}' style='width:100%; max-width:800px;'>\n"


def group_maneuvers(messages, is_active):
  maneuvers: list[tuple[str, list[list]]] = []
  active_prev = False
  description_prev = None

  for msg in messages:
    if msg.which() == 'alertDebug':
      active = is_active(msg.alertDebug.alertText1)
      if active and not active_prev:
        if msg.alertDebug.alertText2 == description_prev:
          maneuvers[-1][1].append([])
        else:
          maneuvers.append((msg.alertDebug.alertText2, [[]]))
        description_prev = maneuvers[-1][0]
      active_prev = active

    if active_prev:
      maneuvers[-1][1][-1].append(msg)

  return maneuvers


def write_report(builder, summary, output_fn):
  sum_idx = builder.index('{ summary }')
  builder[sum_idx:sum_idx + 1] = summary

  with open(output_fn, "w") as f:
    f.write(''.join(builder))

  print(f"\nOpening report: {output_fn}\n")
  webbrowser.open_new_tab(str(output_fn))


def html_table(tabular_data, headers=(), floatfmt="g"):
  rows = [list(row) for row in tabular_data]

  def fmt(val):
    if isinstance(val, str):
      return val
    if isinstance(val, (bool, int)):
      return str(val)
    try:
      return format(val, floatfmt)
    except (TypeError, ValueError):
      return str(val)

  formatted = [[fmt(c) for c in row] for row in rows]
  hdrs = [str(h) for h in headers] if headers else None

  ncols = max((len(r) for r in formatted), default=0)
  if hdrs:
    ncols = max(ncols, len(hdrs))
  if ncols == 0:
    return ""

  for r in formatted:
    r.extend([""] * (ncols - len(r)))
  if hdrs:
    hdrs.extend([""] * (ncols - len(hdrs)))

  parts = ["<table>"]
  if hdrs:
    parts.append("<thead>")
    parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in hdrs) + "</tr>")
    parts.append("</thead>")
  parts.append("<tbody>")
  for row in formatted:
    parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
  parts.append("</tbody>")
  parts.append("</table>")
  return "\n".join(parts)
