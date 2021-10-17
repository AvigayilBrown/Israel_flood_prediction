class Basin:
    """
    This class represents a single basin. 
    """

    # counts the number of basins
    counter = 0

    def __init__(self, id,stations_info, file_flow):
        Basin.counter += 1

        self.id = id

        # basin files
        self.f_flow = file_flow
        self.info = stations_info
        # self.f_prcp = file_prcp

        self.area = None
        self.latitude = None  # m
        self.elevation = None  # m
        
        # dataframe with all basin data
        self.data = None


    def process_station_info(self):
        pass
    # put the area for each basin

    def process_flow_file(self):
        pass
    
    def model(self):
        pass

