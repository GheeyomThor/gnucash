#!/usr/bin/env python3

# python imports
import argparse
import datetime
import functools
import os
import sys
from fractions import Fraction
from itertools import chain
from datetime import date, timedelta
from bisect import bisect_right
import csv

# gnucash imports
from gnucash import Session, GncNumeric, SessionOpenMode, GncPriceDB, GncPrice, GncCommodity, Split

# a dictionary with a period name as key, and number of months in that
# kind of period as the value
PERIODS = {"monthly": 1,
           "quarterly": 3,
           "yearly": 12,
           "daily": None}

NUM_MONTHS = 12

ONE_DAY = timedelta(days=1)

ZERO = Fraction(0)
ONE = Fraction(1)

global gnucash_session
global show_hidden_accounts
global period_length

def gnc_numeric_to_python_decimal(numeric):
    fraction = Fraction(numeric.num(), numeric.denom())
    return fraction

def currency_conversion_factor(gnc_account, date_, target_currency):

    source_commodity = gnc_account.GetCommodity()

    if source_commodity.get_mnemonic() == target_currency.get_mnemonic():
        return ONE
    commodity_in_export_currency = GncPriceDB.lookup_nearest_in_time64(gnucash_session.book.get_price_db(), source_commodity, target_currency, date_)
    if commodity_in_export_currency:
        if GncCommodity(instance=commodity_in_export_currency.get_currency()).get_mnemonic() == target_currency.get_mnemonic():
            factor = gnc_numeric_to_python_decimal(GncNumeric(instance=commodity_in_export_currency.get_value()))
        else:
            factor = 1/gnc_numeric_to_python_decimal(GncNumeric(instance=commodity_in_export_currency.get_value()))
        return factor
    else:
        swig_price_list = GncPriceDB.lookup_nearest_in_time_any_currency_t64(gnucash_session.book.get_price_db(), source_commodity, date_)
        available_prices = [GncPrice(instance=swig_p) for swig_p in swig_price_list]
        if len(available_prices) > 0:
            source_currency_in_export_currency = GncPriceDB.lookup_nearest_in_time64(gnucash_session.book.get_price_db(), available_prices[0].get_currency(), target_currency, date_)
            if not source_currency_in_export_currency:
                print(f"Failed currency conversion for {gnc_account.GetName()} at {date_}")
                currency_conversion = ONE
            else:
                if GncCommodity(instance=source_currency_in_export_currency.get_currency()).get_mnemonic() == target_currency.get_mnemonic():
                    currency_conversion = gnc_numeric_to_python_decimal(GncNumeric(instance=source_currency_in_export_currency.get_value()))
                else:
                    currency_conversion = 1/gnc_numeric_to_python_decimal(GncNumeric(instance=source_currency_in_export_currency.get_value()))

            commodity_in_source_currency = gnc_numeric_to_python_decimal(GncNumeric(instance=available_prices[0].get_value()))
            factor = currency_conversion * commodity_in_source_currency
            return factor
        else:
            print(f"Failed currency conversion for {gnc_account.GetName()} at {date_}")
            return ONE

def next_period_start(start_year, start_month, period_type):
    # add numbers of months for the period length
    end_month = start_month + PERIODS[period_type]
    # use integer division to find out if the new end month is in a different
    # year, what year it is, and what the end month number should be changed
    # to.
    # Because this depends on modular arithmetic, we have to convert the month
    # values from 1-12 to 0-11 by subtracting 1 and putting it back after
    #
    # the really cool part is that this whole thing is implemented without
    # any branching; if end_month > NUM_MONTHS
    #
    # And the super nice thing is that you can add all kinds of period lengths
    # to PERIODS
    end_year = start_year + int((end_month - 1) / NUM_MONTHS)
    end_month = ((end_month - 1) % NUM_MONTHS) + 1

    return end_year, end_month


def period_end(start_year, start_month, period_type):  # next_period_start - 1
    if period_type not in PERIODS:
        raise Exception("%s is not a valid period, should be %s" % (period_type, str(list(PERIODS.keys()))))

    end_year, end_month = next_period_start(start_year, start_month, period_type)

    # last step, the end date is day back from the start of the next period
    # so we get a period end like
    # 2010-03-31 for period starting 2010-01 instead of 2010-04-01
    return date(int(end_year), int(end_month), 1) #- ONE_DAY


def generate_period_boundaries(start_year, start_month, period_type, now):
    now_year = now.year
    now_month = now.month
    if period_type == "daily":
        now_day = now.day
        start_day = now_day
        while start_year < now_year or (start_year == now_year and (start_month < now_month or (start_month == now_month and start_day < now_day))):
            d = date(int(start_year), int(start_month), start_day)
            next_day = d + ONE_DAY
            yield d, next_day
            start_year, start_month, start_day = next_day.year, next_day.month, next_day.day

    else:
        while start_year < now_year or (start_year == now_year and start_month <= now_month):
            yield date(int(start_year), int(start_month), 1), period_end(start_year, start_month, period_type)
            start_year, start_month = next_period_start(start_year, start_month, period_type)

def account_from_path(top_account, account_path, original_path=None):
    """ finds an account below top_account of name account_path[-1] within the path hints given in the account_path[:-1] """
    if original_path is None:
        original_path = account_path
    account, account_path = account_path[0], account_path[1:]

    account = top_account.lookup_by_name(account)
    if account is None:
        raise Exception(
            "path '" + '::..::'.join(original_path) + "' could not be found")
    if len(account_path) > 0:
        return account_from_path(account, account_path, original_path)
    else:
        return account

# see https://github.com/Gnucash/gnucash/blob/maint/bindings/python/example_scripts/export_account_totals.py
def get_all_sub_accounts(account, names=[], depth=1):
    """Iterate over all sub account of a given account."""
    # and (len(acc.get_children()) > 0 or gnc_numeric_to_python_decimal(acc.GetBalance()) > 0)
    for child in filter(lambda acc: show_hidden_accounts or not acc.IsHidden(), account.get_children_sorted()):
        child_names = names.copy()
        child_names.append(child.GetName())
        yield child, '::'.join(child_names), depth
        yield from get_all_sub_accounts(child, child_names, depth+1)


def get_parent_and_all_sub_accounts(parent_account):
    """Yields (account, account name, account depth to parent)"""
    yield parent_account, parent_account.GetName(), 0
    yield from get_all_sub_accounts(parent_account, [parent_account.GetName()])

def main():
    """https://github.com/Gnucash/gnucash/
    https://wiki.gnucash.org/wiki/Python_Bindings"""

    parser = argparse.ArgumentParser(description='''
            Processes .gnucash file and returns a .csv file containing amount, gain, yield and their averages per period and in total for a given account and its children.
        ''',
                                     epilog='''
            Example:
            python3 ./main.py --gnucash_file /my/path/to/file.gnucash --year 1996 --month 1 --period_type yearly --reduction_depth 1 --currency GBP --show_hidden --account_path "Big Bank"
        ''')
    parser.add_argument("--gnucash_file", help="data source .gnucash file path", required=True)
    parser.add_argument("--year", type=int, help="start year", required=True)
    parser.add_argument("--month", type=int, help="start month", required=True)
    parser.add_argument("--period_type", help="accumulation period length can be monthly, quarterly or yearly", required=True)
    parser.add_argument("--reduction_depth", type=int, help="depth from the account name specified at which all children results will be accumulated under", required=True)
    parser.add_argument("--currency", help="Export currency. If not specified, the account currency is used.", required=False)
    parser.add_argument("--show_hidden", action='store_true', help="Do hidden account need processing?", required=False)
    parser.add_argument('--account_path', nargs='+', help='Blank separated account path hints down to the name of the account of interest', required=True)
    args = parser.parse_args()

    export_path = f"{os.path.abspath(os.path.dirname(sys.argv[0]))}/test_scripts"

    gnucash_file = args.gnucash_file
    start_year = args.year
    start_month = args.month
    period_type = args.period_type
    reduction_depth = args.reduction_depth
    currency = args.currency
    show_hidden = args.show_hidden
    account_path = args.account_path

    # start_year, start_month = [int(blah) for blah in (start_year, start_month)] # Start year and month
    # reduction_depth = int(reduction_depth)  # Depth from the root account at which subAccounts are reduced into their parent
    # show_hidden = show_hidden == 'True' # Show gnucash hidden accounts?
    # account_path = argv[8:]     # Path hint of the root account to analyse, space separated, starting after the 'Assets' account path

    total_rows = generate(export_path, gnucash_file, account_path, start_year, start_month, period_type, reduction_depth, currency, show_hidden)

    print(total_rows)


def generate(export_path, gnucash_file, account_path, start_year=1996, start_month=1, period_type="yearly", reduction_depth=1, currency=None, show_hidden=False):

    now = datetime.datetime.now() + ONE_DAY
    global period_length
    period_length = period_type

    global show_hidden_accounts
    show_hidden_accounts = show_hidden

    try:
        global gnucash_session
        gnucash_session = Session(gnucash_file, SessionOpenMode.SESSION_NORMAL_OPEN)
        root_account = gnucash_session.book.get_root_account()
        asset_root_account = root_account.lookup_by_name("Assets")
        parent_asset_account = account_from_path(root_account, ["Assets"] + account_path)
        income_root_account = root_account.lookup_by_name("Income")
        expenses_root_account = root_account.lookup_by_name("Expenses")
        export_currency = gnucash_session.book.get_table().lookup('ISO4217', currency) if currency else None

        # a list of all the periods of interest, for each period
        # keep the start date, end date, a list to store debits_list and credits_list,
        # and sums for tracking the sum of all debits_list and sum of all credits_list
        all_sub_asset_accounts = [acc for acc in get_parent_and_all_sub_accounts(parent_asset_account)]
        period_list = [
            {
                "start_date": start_date, "end_date": end_date,
                "accounts": [
                    {
                        "child_account_name": child_account[1],
                        "debits_list": [],  # debits_list
                        "credits_list": [],  # credits_list
                        "debits_sums": ZERO,  # debits_list sum
                        "credits_sums": ZERO,  # credits_list sum
                        "asset_value_end": ZERO,
                        "asset_value_start": ZERO,
                        "asset_in": ZERO,
                        "asset_out": ZERO,
                        "unreal_reinvested": ZERO,

                    }
                    for child_account in all_sub_asset_accounts
                ]
            }
            for start_date, end_date in generate_period_boundaries(start_year, start_month, period_length, now)
        ]
        # a copy of the above list with just the period start dates
        period_starts = [period["start_date"] for period in period_list]

        for asset_child_account in all_sub_asset_accounts:

            # Export currency
            if export_currency:
                asset_target_currency = export_currency
            else:
                parent_with_currency = None
                j = 1
                while not parent_with_currency:
                    parent_with_currency = next(filter(lambda acc: acc[0].GetCommodity().is_currency() and
                                                                   (acc[1] == asset_child_account[1] or acc[1] == "::".join(asset_child_account[1].split('::')[:-j]) and
                                                                    acc[2] <= reduction_depth),
                                                       all_sub_asset_accounts), None)
                    j += 1
                asset_target_currency = parent_with_currency[0].GetCommodity()

            # ins, outs, unrealized
            for asset_split in asset_child_account[0].GetSplitList():
                trans = asset_split.parent
                trans_date = trans.GetDate().date()
                period_index = bisect_right(period_starts, trans_date) - 1
                if period_index >= 0 and trans_date < period_list[len(period_list) - 1]['end_date']:

                    # Period
                    period = period_list[period_index]
                    assert period["start_date"] <= trans_date < period["end_date"], f'!{period["start_date"]} <= {trans_date} < {period["end_date"]}'
                    period_account = [acc for acc in period["accounts"] if acc["child_account_name"] == asset_child_account[1]][0]

                    # Amount
                    # trans_acc_value = trans.GetAccountValue(asset_child_account[0])
                    # split_amount = currency_conversion_factor(??, trans_date) * gnc_numeric_to_python_decimal(trans_acc_value)
                    if not asset_child_account[0].GetCommodity().is_currency():
                        amount_b = Fraction(0)  # Decimal(0)
                        for pay_acc_split in trans.GetPaymentAcctSplitList():
                            pay_acc_split = Split(instance=pay_acc_split)
                            if pay_acc_split.GetAmount().positive_p() != asset_split.GetAmount().positive_p():
                                amount_b += currency_conversion_factor(pay_acc_split.GetAccount(), trans_date, asset_target_currency) * gnc_numeric_to_python_decimal(
                                    pay_acc_split.GetAmount())
                        split_amount = -amount_b
                    else:
                        split_amount = currency_conversion_factor(asset_child_account[0], trans_date, asset_target_currency) * gnc_numeric_to_python_decimal(
                            asset_split.GetAmount())

                    if split_amount < 0:
                        period_account["asset_out"] -= split_amount  # split is - (removed from asset)
                    else:
                        period_account["asset_in"] += split_amount  # split is + in (added to asset)

                    # Unreal_reinvested
                    trans_splits = [s for s in trans.GetSplitList()]
                    for trans_split in trans_splits:
                        try:
                            trans_split_acc = trans_split.GetAccount()
                            if trans_split_acc.GetType() == 8:  # Split amount is from an Income account
                                # Check the income account path
                                income_child_account = account_from_path(root_account, ["Income"] + account_path[:-1] + asset_child_account[1].split('::'))
                                all_income_sub_accounts_names = [acc[1].split("::")[-1] for acc in get_parent_and_all_sub_accounts(income_child_account)]
                                next(filter(lambda income_acc: income_acc == trans_split_acc.GetName(), all_income_sub_accounts_names))
                                print(
                                    f"Added Reinvested income {trans_split.GetAmount()} into {asset_child_account[0].GetName()} at {trans_date} from Income account {trans_split_acc.GetName()}")

                                # trans_split.GetAccount() split is - (removed from Income)
                                period_account["unreal_reinvested"] += -currency_conversion_factor(trans_split_acc, trans_date,
                                                                                                   asset_target_currency) * gnc_numeric_to_python_decimal(trans_split.GetAmount())
                        except Exception as e:
                            print(
                                f"No Reinvested income added {trans_split.GetAmount()} into '{asset_child_account[0].GetName()}' at {trans_date}: No Income account '{trans_split_acc.GetName()}' under {asset_child_account[0].GetName()}: {repr(e)}")

            # insert and add all splits in the periods of interest
            try:
                income_child_account = account_from_path(income_root_account, account_path[:-1] + asset_child_account[1].split('::'))
            except:
                income_child_account = None
            if income_child_account:
                for income_split in income_child_account.GetSplitList():
                    trans = income_split.parent
                    trans_date = trans.GetDate().date()
                    period_index = bisect_right(period_starts, trans_date) - 1
                    if period_index >= 0 and trans_date < period_list[len(period_list) - 1]['end_date']:
                        period = period_list[period_index]
                        assert period["start_date"] <= trans_date < period["end_date"], f'!{period["start_date"]} <= {trans_date} < {period["end_date"]}'
                        split_amount = currency_conversion_factor(income_child_account, trans_date, asset_target_currency) * gnc_numeric_to_python_decimal(income_split.GetAmount())

                        period_account = [acc for acc in period["accounts"] if acc["child_account_name"] == asset_child_account[1]][0]
                        period_account["credits_list"].append((trans, income_split))
                        period_account["credits_sums"] -= split_amount  # split is - income

            try:
                expenses_child_account = account_from_path(expenses_root_account, account_path[:-1] + asset_child_account[1].split('::'))
            except:
                expenses_child_account = None
            if expenses_child_account:
                for expenses_split in expenses_child_account.GetSplitList():
                    trans = expenses_split.parent
                    trans_date = trans.GetDate().date()
                    period_index = bisect_right(period_starts, trans_date) - 1
                    if period_index >= 0 and trans_date < period_list[len(period_list) - 1]['end_date']:
                        period = period_list[period_index]
                        assert (period["start_date"] <= trans_date < period["end_date"])
                        split_amount = currency_conversion_factor(expenses_child_account, trans_date, asset_target_currency) * gnc_numeric_to_python_decimal(
                            expenses_split.GetAmount())
                        period_account = [acc for acc in period["accounts"] if acc["child_account_name"] == asset_child_account[1]][0]
                        period_account["debits_list"].append((trans, expenses_split))
                        period_account["debits_sums"] += split_amount  # split is + in expenses

            for period in period_list:
                period_account = [acc for acc in period["accounts"] if acc["child_account_name"] == asset_child_account[1]][0]

                start_balance = gnc_numeric_to_python_decimal(asset_child_account[0].GetBalanceAsOfDate(period["start_date"]))
                period_account["asset_value_start"] = currency_conversion_factor(asset_child_account[0], period['start_date'], asset_target_currency) * start_balance

                next_period_index = bisect_right(period_starts, period['start_date'])
                next_period_start_date = period_list[next_period_index]["start_date"] if next_period_index < len(period_list) else now
                end_balance = gnc_numeric_to_python_decimal(asset_child_account[0].GetBalanceAsOfDate(next_period_start_date))
                period_account["asset_value_end"] = currency_conversion_factor(asset_child_account[0], period['end_date'], asset_target_currency) * end_balance

        # add dead incomes and expenses
        try:
            parent_income_account = account_from_path(root_account, ["Income"] + account_path)
            all_income_sub_accounts = [acc for acc in get_parent_and_all_sub_accounts(parent_income_account)]
            for income_sub_account in all_income_sub_accounts:
                try:
                    asset_child_account = account_from_path(asset_root_account, account_path[:-1] + income_sub_account[1].split('::'))
                except:
                    asset_child_account = None
                if not asset_child_account:  # dead income without a matching asset
                    print(f"Dead (wo matching asset account) income account: {income_sub_account[1]}", end=' ')
                    next_valid_parent = None
                    i = 1
                    while not next_valid_parent and i <= len(income_sub_account[1].split('::')):
                        next_valid_parent = next(filter(lambda acc: acc[1] == "::".join(income_sub_account[1].split('::')[:-i]), all_sub_asset_accounts), None)
                        i += 1
                    if not next_valid_parent:
                        next_valid_parent = (income_root_account, "Income", 0)
                    print(f"=> {next_valid_parent[1]}")
                    for income_split in income_sub_account[0].GetSplitList():
                        trans = income_split.parent
                        trans_date = trans.GetDate().date()
                        period_index = bisect_right(period_starts, trans_date) - 1
                        if period_index >= 0 and trans_date < period_list[len(period_list) - 1]['end_date']:
                            period = period_list[period_index]
                            assert (period["start_date"] <= trans_date < period["end_date"])
                            split_amount = currency_conversion_factor(next_valid_parent[0], trans_date, asset_target_currency) * gnc_numeric_to_python_decimal(
                                income_split.GetAmount())

                            period_account = [acc for acc in period["accounts"] if acc["child_account_name"] == next_valid_parent[1]][0]
                            period_account["credits_list"].append((trans, income_split))
                            period_account["credits_sums"] -= split_amount  # split is - income
        except:
            print(f"{account_path} has no income")
        try:
            parent_expense_account = account_from_path(root_account, ["Expenses"] + account_path)
            all_expense_sub_accounts = [acc for acc in get_parent_and_all_sub_accounts(parent_expense_account)]
            for expense_sub_account in all_expense_sub_accounts:
                try:
                    asset_child_account = account_from_path(asset_root_account, account_path[:-1] + expense_sub_account[1].split('::'))
                except:
                    asset_child_account = None
                if not asset_child_account:  # dead income without a matching asset
                    print(f"Dead (wo matching asset account) expenses account: {expense_sub_account[1]}", end=' ')
                    next_valid_parent = None
                    i = 1
                    while not next_valid_parent and i <= len(income_sub_account[1].split('::')):
                        next_valid_parent = next(filter(lambda acc: acc[1] == "::".join(expense_sub_account[1].split('::')[:-i]), all_sub_asset_accounts), None)
                        i += 1
                    if not next_valid_parent:
                        next_valid_parent = (expenses_root_account, "Expenses", 0)
                    print(f"=> {next_valid_parent[1]}")
                    for expense_split in expense_sub_account[0].GetSplitList():
                        trans = expense_split.parent
                        trans_date = trans.GetDate().date()
                        period_index = bisect_right(period_starts, trans_date) - 1
                        if period_index >= 0 and trans_date < period_list[len(period_list) - 1]['end_date']:
                            period = period_list[period_index]
                            assert (period["start_date"] <= trans_date < period["end_date"])
                            split_amount = currency_conversion_factor(next_valid_parent[0], trans_date, asset_target_currency) * gnc_numeric_to_python_decimal(
                                expense_split.GetAmount())

                            period_account = [acc for acc in period["accounts"] if acc["child_account_name"] == next_valid_parent[1]][0]
                            period_account["debits_list"].append((trans, expense_split))
                            period_account["debits_sums"] += split_amount  # split is + in expenses
        except:
            print(f"{account_path} has no expenses")

        # reduce to reduction_depth
        deepest_sub_accounts = list(filter(lambda acc: acc[2] == reduction_depth, all_sub_asset_accounts))
        reduced_sub_accounts = list(filter(lambda acc: acc[2] <= reduction_depth, all_sub_asset_accounts))

        def deepish_period_list_copy(period):
            account_deepish_copy = list(map(lambda acc: {
                "child_account_name": acc["child_account_name"],
                "debits_list": acc["debits_list"].copy(),  # debits_list
                "credits_list": acc["credits_list"].copy(),  # credits_list
                "debits_sums": acc["debits_sums"],  # debits_list sum
                "credits_sums": acc["credits_sums"],  # credits_list sum
                "asset_value_end": acc["asset_value_end"],
                "asset_value_start": acc["asset_value_start"],
                "asset_in": acc["asset_in"],
                "asset_out": acc["asset_out"],
                "unreal_reinvested": acc["unreal_reinvested"]
            }, period["accounts"]))

            return {  # deepish copy
                "start_date": period["start_date"], "end_date": period["end_date"],
                "accounts": account_deepish_copy
            }

        reduced_period_list = list(map(deepish_period_list_copy, period_list))

        def sum_acc(accumulator, period_acc_element):
            accumulator["debits_sums"] += period_acc_element["debits_sums"]
            accumulator["credits_sums"] += period_acc_element["credits_sums"]
            accumulator["debits_list"] += period_acc_element["debits_list"]
            accumulator["credits_list"] += period_acc_element["credits_list"]
            accumulator["asset_value_end"] += period_acc_element["asset_value_end"]
            accumulator["asset_value_start"] += period_acc_element["asset_value_start"]
            accumulator["asset_in"] += period_acc_element["asset_in"]
            accumulator["asset_out"] += period_acc_element["asset_out"]
            accumulator["unreal_reinvested"] += period_acc_element["unreal_reinvested"]
            return accumulator

        for deepest_account in deepest_sub_accounts:
            deepest_account_value = deepest_account[0]
            deepest_account_name = deepest_account[1]
            out_off_depth_accounts_names = list(map(lambda acc: f'{deepest_account_name}::{acc[1]}', get_all_sub_accounts(deepest_account_value, depth=reduction_depth + 1)))
            for period_row in reduced_period_list:
                deepest_account_period_row = list(filter(lambda acc: acc["child_account_name"] == deepest_account_name, period_row["accounts"]))[0]
                out_off_depth_accounts_period = list(filter(lambda p_acc: p_acc["child_account_name"] in out_off_depth_accounts_names, period_row["accounts"]))
                reduced_period_row = functools.reduce(sum_acc, out_off_depth_accounts_period, deepest_account_period_row)
                for p_acc1 in out_off_depth_accounts_period:
                    period_row["accounts"].remove(p_acc1)
                period_row["accounts"][period_row["accounts"].index(deepest_account_period_row)] = reduced_period_row

        # printing out
        print_out_csv(all_sub_asset_accounts, True, True, period_list,
                      open(f"{export_path}/exports/gain_yield_raw_{'::'.join(account_path)}_{start_year}_{start_month}_{period_length}_{reduction_depth}_{currency}_{show_hidden}.csv", "w"))
        total_row = print_out_csv(reduced_sub_accounts, False, False, reduced_period_list,
                                  open(f"{export_path}/exports/gain_yield_reduced_{'::'.join(account_path)}_{start_year}_{start_month}_{period_length}_{reduction_depth}_{currency}_{show_hidden}.csv", "w"))

        # no save needed, we're just reading..
        gnucash_session.end()

        return total_row

    except Exception as e:
        print(e)
        if "gnucash_session" in globals():
            gnucash_session.end()
        raise e


def print_out_csv(sub_accounts, credits_show, debits_show, period_list, file):
    csv_writer = csv.writer(file)
    children_header = list(chain.from_iterable(map(
        lambda acc: [f'{acc[1].split("::")[-1]} Asset', f'Gain', f'Gain Rate', f'Income', f'Expenses', f'Yield', f'Total Yield {acc[1].split("::")[-1]}'], sub_accounts
    )))
    csv_writer.writerow(['period start', 'period end'] + children_header)

    def child_lists_row_details(acc, values):
        return (('', '', '', '', trans.GetDescription(), currency_conversion_factor(acc, trans.GetDate().date(), acc.GetCommodity()) * gnc_numeric_to_python_decimal(split.GetAmount()))
                for trans, split in values)

    def account_sums_row_details(period_account):
        asset_value_end = period_account["asset_value_end"]
        asset_value_start = period_account["asset_value_start"]
        moneys_out = period_account["asset_out"]
        moneys_in = period_account["asset_in"]
        unreal_reinvested = period_account["unreal_reinvested"]
        period_credits = period_account["credits_sums"]
        period_debits = period_account["debits_sums"]
        transactions = moneys_out - moneys_in
        base = (asset_value_start - transactions)
        gain = (asset_value_end + unreal_reinvested) - base
        income = period_credits - unreal_reinvested
        gain_rate = 0 if base == 0 else gain / base
        period_yield = 0 if base == 0 else (income - period_debits) / base
        return [
            f'{float(asset_value_end) :.2f}',  # Asset
            f'{float(gain) :.2f}',  # Gain: value change + out - in + unreal(from reinvested income)
            f'\'{float(gain_rate) :.3%}',  # Gain rate: gain/asset value at start
            f'{float(income) :.2f}',  # Income: period_credits - unreal
            f'{float(period_debits) :.2f}',  # Expenses
            f'\'{float(period_yield) :.3%}',   # Yield: (Income - period_debits))/asset value at end
            f'\'{float(gain_rate + period_yield) :.3%}'
        ]

    asset_sums = {acc[1]: ZERO for acc in sub_accounts}
    gain_sums = {acc[1]: ZERO for acc in sub_accounts}
    nb_asset_period = {acc[1]: 0 for acc in sub_accounts}

    credits_sums = {acc[1]: ZERO for acc in sub_accounts}
    debits_sums = {acc[1]: ZERO for acc in sub_accounts}

    for period_row in period_list:

        children_sums_row = list(chain.from_iterable(map(account_sums_row_details, period_row["accounts"])))
        csv_writer.writerow([period_row["start_date"], period_row["end_date"]] + children_sums_row)

        for period_acc in period_row["accounts"]:

            period_asset = period_acc["asset_value_end"]
            asset_sums[period_acc["child_account_name"]] += period_asset
            if period_asset != 0:
                nb_asset_period[period_acc["child_account_name"]] += 1

            period_gain = ((period_asset - period_acc["asset_value_start"]) + (period_acc["asset_out"] - period_acc["asset_in"]) + period_acc["unreal_reinvested"])
            gain_sums[period_acc["child_account_name"]] += period_gain

            period_income = (period_acc["credits_sums"] - period_acc["unreal_reinvested"])
            credits_sums[period_acc["child_account_name"]] += period_income
            debits_sums[period_acc["child_account_name"]] += period_acc["debits_sums"]

            sub_acc = next(filter(lambda sa: period_acc["child_account_name"] == sa[1], sub_accounts))
            if debits_show and len(period_acc["debits_list"]) > 0:
                csv_writer.writerow(('DEBITS', '', '', '', 'description', 'value'))
                csv_writer.writerows(child_lists_row_details(sub_acc[0], period_acc["debits_list"]))
                csv_writer.writerow(())
            if credits_show and len(period_acc["credits_list"]) > 0:
                csv_writer.writerow(('CREDITS', '', '', '', 'description', 'value'))
                csv_writer.writerows(child_lists_row_details(sub_acc[0], period_acc["credits_list"]))
                csv_writer.writerow(())

    csv_writer.writerow([f'Averages {period_length}:'])
    # Asset', f'Gain', f'Gain Rate', f'Income', f'Expenses', f'Yield', 'Total Yield'

    totals = list(map(lambda acc_stats: [f"{acc_stats[5]}",
                                         f"{0 if acc_stats[0] == 0 else float(acc_stats[2] / acc_stats[0]):.2f}",  # Gain avg
                                         f"{0 if acc_stats[1] == 0 else float(acc_stats[2] / acc_stats[1]):.3%}",  # Gain rate
                                         f"{0 if acc_stats[0] == 0 else float(acc_stats[3] / acc_stats[0]):.2f}",  # Credits avg
                                         f"{0 if acc_stats[0] == 0 else float(acc_stats[4] / acc_stats[0]):.2f}",  # Debits avg
                                         f"{0 if acc_stats[1] == 0 else float((acc_stats[3] - acc_stats[4]) / acc_stats[1]):.3%}",  # Yield
                                         f"{0 if acc_stats[1] == 0 else float((acc_stats[2] + (acc_stats[3] - acc_stats[4])) / acc_stats[1]):.3%}"],  # Total Yield
                 zip(nb_asset_period.values(), asset_sums.values(), gain_sums.values(), credits_sums.values(), debits_sums.values(), nb_asset_period.keys())))

    children_totals_row = list(chain.from_iterable(totals))
    csv_writer.writerow([period_list[0]["start_date"], period_list[-1]["end_date"]] + children_totals_row)

    return [a + b for a, b in zip(totals, [[float(0 if a[0] == 0 else a[1]/a[0])] for a in zip(nb_asset_period.values(), asset_sums.values())])]

if __name__ == "__main__":
    main()
