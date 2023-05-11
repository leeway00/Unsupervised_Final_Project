# Some financial ratios to add to the dataset

# Price-Earnings (P/E) Ratio
df['pe'] = df['close'] / (df['net_income_loss_share_holder'] / df['tot_share_holder_equity'])

# Price-To-Book (P/B) Ratio
df['pb'] = df['close'] / (df['tot_share_holder_equity'] / df['tot_share_holder_equity'])

# Price-To-Sales (P/S) Ratio
df['ps'] = df['close'] / (df['tot_revnu'] / df['tot_share_holder_equity'])

# Debt-To-Equity (D/E) Ratio
df['de'] = (df['tot_curr_liab'] + df['tot_lterm_debt']) / df['tot_share_holder_equity']

# Earnings Yield
df['ey'] = df['net_income_loss_share_holder'] / df['close']

# Current Ratio
df['cr'] = df['tot_curr_asset'] / df['tot_curr_liab']

# Quick Ratio (Acid-Test Ratio)
df['qr'] = (df['tot_curr_asset'] - df['invty']) / df['tot_curr_liab']

# Inventory Turnover
df['it'] = df['cost_good_sold'] / df['invty']

# Asset Turnover
df['at'] = df['tot_revnu'] / df['tot_asset']

# Return on Assets (ROA)
df['roa'] = df['net_income_loss'] / df['tot_asset']

# Interest Coverage Ratio
df['ic'] = df['ebit'] / df['int_exp_oper']