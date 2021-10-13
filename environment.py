class Environment:
    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data # 2차원배열
        self.observation = None # 차트데이터의 한줄
        self.idx = -1 #차트데이터의 현재 위치 

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self): #Learners.build_sample()에 쓰임.
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self): # observe종가 가져오기| Agent.get_states/action/validate_action에 쓰임
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data): #코드가 안쓰임
        self.chart_data = chart_data
