import baostock as bs
import pandas as pd
import os





def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='1990-01-01',
                 date_end='2024-03-16'):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,open,high,low,close,amount,volume"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date(self.date_end)
        for index, row in stock_df.iterrows():
            print(f'processing {row["code"]} {row["code_name"]}')
            df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                               start_date=self.date_start,
                                               end_date=self.date_end).get_data()
        # 替换文件名中的*字符为_
            code_name = row["code_name"].replace('*', '_')
            df_code.to_csv(f'{self.output_dir}/{row["code"]}.{code_name}.csv', index=False)
        self.exit()



if __name__ == '__main__':
    # 获取全部股票的日K线数据
    mkdir('./data/train')
    downloader = Downloader('./data/train', date_start='2000-01-01', date_end='2024-2-29')
    downloader.run()

    mkdir('./data/test')
    downloader = Downloader('./data/test', date_start='2024-3-1', date_end='2024-3-15')
    downloader.run()

