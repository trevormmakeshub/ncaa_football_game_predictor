Run python scripts/refresh_today.py during the season. It does not run on a timer.

FBS teams come from the ESPN standings tree, including division children such as the Sun Belt. Each team schedule is requested for 2025 and 2026, and again for postseason games. A final score is stored only when ESPN marks the game STATUS_FINAL, the Eastern date is on or before 2026-11-20, and the score is not 0-0. Postponed games are not stored as finals. Schedule request failures: 0. Games against non-FBS opponents stay on the FBS schedule. A non-FBS team's prior form includes only the games that appear on those schedules.

Form uses only earlier games: win rate, points scored, points allowed, rest days, and the last four games when that check improves the holdout. A game does not train on its own score. Ties are stored and are left out of the win classifier. The published home_win_prob on a completed game is in-sample. The held-out accuracy below is a separate fit on games from 2026-08-01. The win model is a ridge classifier. Points use linear regression. A half-weight of prior points per game, or a residual on top of prior points per game, is used only when that choice lowers the holdout total error. Opponent win rate is the other side's prior win rate.

Ranking rows: 1526. Weeks are the occurrence on each poll. Player rows are the ESPN 2026 regular-season leader table. The site rankings endpoint ignores the requested week and returns the latest poll, so it is not used for past weeks. There is no recorded public closing line in this table. spread_line and total_line are blank. The line column says this is not a bet. This does not price a bet.

FBS teams: 138
Completed rows: 1716
2025 regular-season games: 912
2025 postseason games: 46
2026 regular-season games: 758
Ties: 0
Zero-score rows stored: 0
Training rows: 1562
Feature count: 12
Minimum date: 2025-08-23
Maximum date: 2026-11-19
Next games through 2026-11-27: 80
Next games dated 2026-11-20: 6
Next games with home_win_prob: 80
Held-out games from 2026-08-01: 749 of 813 training rows, accuracy 0.7263
In-sample accuracy: 0.7113 on 1562 scored rows
High-confidence next games: 21
Baseline holdout: accuracy 0.7196, total MAE 15.686, prior-points blend MAE 14.394, residual MAE 15.686, 749 test games after 813 earlier games
Recent-form holdout: accuracy 0.7263, total MAE 15.683, prior-points blend MAE 14.420, residual MAE 15.683, 749 test games after 813 earlier games
Recent form adopted: yes
Points blend adopted: yes
Points method: blend
Current-season team rows: 138
Player rows: 179
Checked bowl: 2026-01-01 MISS 39 UGA 34
