#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:43:41 2023

@author: dimitrios
"""

import oandapyV20
import oandapyV20.endpoints.orders as orders
from oandapyV20.endpoints.orders import OrderDetails, OrderCancel
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.trades import TradeClose

class OandaAPI:
    def __init__(self, api_token, account_id):
        self.api_token = api_token
        self.account_id = account_id
        self.api = oandapyV20.API(access_token=api_token)
    
    def place_order(self, order_type, instrument, units, stop_loss=None, take_profit=None):
        """
        Place an order using the OANDA API.
    
        Parameters:
            order_type (str): Order type, either "MARKET" for market orders or "LIMIT" for limit orders.
            instrument (str): Currency pair instrument.
            units (int): Number of units to buy or sell.
            stop_loss (float, optional): Stop loss price. Default is None.
            take_profit (float, optional): Take profit price. Default is None.
    
        Returns:
            dict: Order response from the API.
        """
        order_params = {
            "order": {
                "units": str(units),
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": order_type,
                "positionFill": "DEFAULT"
            }
        }
    
        if stop_loss is not None:
            order_params["order"]["stopLossOnFill"] = {"price": str(stop_loss)[:7]}
    
        if take_profit is not None:
            order_params["order"]["takeProfitOnFill"] = {"price": str(take_profit)[:7]}
    
        order_request = orders.OrderCreate(accountID=self.account_id, data=order_params)
        self.api.request(order_request)
    
        return order_request.response

    def close_order(self, order_id):
        """
        Close an existing order using the OANDA API.

        Parameters:
            order_id (str): ID of the order to be closed.

        Returns:
            dict: Order cancellation response from the API.
        """
        order_cancellation_request = OrderCancel(self.account_id, order_id)
        order_cancellation_response = self.api.request(order_cancellation_request)
    
        return order_cancellation_response



    def close_trade(self, trade_id):
        """
        Close an existing trade using the OANDA API.
        
        Parameters:
            trade_id (str): ID of the trade to be closed.
        
        Returns:
            dict: Trade close response from the API.
        """
        try:
            trade_close_request = TradeClose(self.account_id, trade_id)
            trade_close_response = self.api.request(trade_close_request)
            return trade_close_response
        except V20Error as e:
            if "TRADE_DOESNT_EXIST" in str(e):
                print(f"Trade {trade_id} does not exist. Skipping close.")
                return None
            else:
                raise e

