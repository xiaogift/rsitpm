#!/usr/bin/env python3
# ============================================================================== #
# Recommender System Wide & Deep Model @ IT Employees Data For Project Allocation
# https://www.kaggle.com/datasets/granjithkumar/it-employees-data-for-project-allocation
# Powered by xiaolis@outlook.com 202307
# ============================================================================== #
import os, torch, numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pandas import read_csv, notnull
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================== #
class EmployeeData:

    def __init__(self, root='./data'):
        employee = read_csv( root+'/Final_Employees_Data.csv', 
                          sep = ',', header = None,
                          names = [ 'Eid','Ename','Experience','Total_projects','Rating','Area_of_Interest_1',\
                                    'Area_of_Interest_2','Area_of_Interest_3','Language1','Language2','Language3',\
                                    'AI_project_count','ML_project_count','JS_project_count','Java_project_count',\
                                    'DotNet_project_count','Mobile_project_count'] ).drop(0)
        skills = read_csv( root+'/Employee_Skills_Datset.csv', 
                            sep = ',', header = None, 
                            names = [ 'Eid','Python','Machine Learning','Deep Learning','Data Analysis','Asp.Net',\
                                      'Ado.Net','VB.Net','C#','Java','Spring Boot','Hibernate','NLP','CV','JS','React',\
                                      'Node','Angular','Dart','Flutter','Vb.Net'] ).drop(0)
        self.data = employee.merge(skills, on='Eid', how='left')
        self.le, self.ss = LabelEncoder(), StandardScaler()
        self._process_wide()
        self._process_deep()

    def _process_wide(self):
        cols_ints = ['Area_of_Interest_1','Area_of_Interest_2','Area_of_Interest_3']
        cols_lang = ['Language1','Language2','Language3']
        cols_prjs = ['AI_project_count','ML_project_count','JS_project_count','Java_project_count',\
                     'DotNet_project_count','Mobile_project_count']
        cols_skil = ['Python','Machine Learning','Deep Learning','Data Analysis','Asp.Net',\
                     'Ado.Net','VB.Net','C#','Java','Spring Boot','Hibernate','NLP','CV','JS','React',\
                     'Node','Angular','Dart','Flutter','Vb.Net']
        self.data['Projects_count'] = self._multi_hot(cols_prjs)
        for c in cols_ints+cols_lang+cols_skil: self.data[c] = self.le.fit_transform(self.data[c])
        self.data['Interests'] = self._multi_hot(cols_ints)
        self.data['Languages'] = self._multi_hot(cols_lang)
        self.data['Skills'] = self._multi_hot(cols_skil)

    def _process_deep(self):
        self.data['Firstname'] = self.le.fit_transform(self.data['Ename'].str.split().str[0])
        self.data['Surname'] = self.le.fit_transform(self.data['Ename'].str.split().str[-1])
        for c in ['Firstname','Surname']: self.data[c] = self.le.fit_transform(self.data[c])
        self.data.drop('Ename', axis=1, inplace=True)
        for c in ['Firstname','Surname', 'Total_projects', 'Experience']: 
            self.data[c] = self.ss.fit_transform(self.data[c].values.reshape(-1, 1))
        self.data['Rating'] = self.data['Rating'].values.astype(float)
        self.data['Eid'] = self.data['Eid'].values.astype(float)

    def _multi_hot(self, cols):
        res = self.data[cols].apply(lambda row: np.array(row.values), axis=1)
        self.data.drop(cols, axis=1, inplace=True)
        return res

    def get_trn_val(self): 
        return train_test_split(self.data, test_size=0.3, random_state=42)

class EmployeeDataset(Dataset):
    def __len__(self): return len(self.data)
    def __init__(self, data, wide_cols, deep_cols): 
        self.data = data
        self.col_wide = wide_cols
        self.col_deep = deep_cols
    def __getitem__(self, index):
        current_row = self.data.iloc[index]
        if 'Eid' in self.col_wide:
            wide = torch.cat([torch.tensor(current_row[c].astype(np.float32), dtype=torch.float) for c in self.col_wide[1:]], dim=0)
            wide = torch.cat((torch.tensor(current_row['Eid'], dtype=torch.float).unsqueeze(0), wide), dim=0)
        else: wide = torch.cat([torch.tensor(current_row[c].astype(np.float32), dtype=torch.float) for c in self.col_wide], dim=0)
        return { "wide_inputs": wide,
                 "deep_inputs": torch.tensor(current_row[self.col_deep].values.astype(np.float32), dtype=torch.float),
                 "rating": torch.tensor(current_row['Rating'], dtype=torch.float) }

def get_dataloader(col_wide, col_deep):
    trn_data, val_data = EmployeeData().get_trn_val()
    trn_dataloader = DataLoader( EmployeeDataset(trn_data,col_wide, col_deep), batch_size = 100, shuffle = True )
    val_dataloader = DataLoader( EmployeeDataset(val_data,col_wide, col_deep), batch_size = 100, shuffle = False )
    return trn_dataloader, val_dataloader

# ============================================================================== #
class WideAndDeep(torch.nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_dim, output_dim):
        super(WideAndDeep, self).__init__()
        self.wide_linear = torch.nn.Linear(wide_dim, output_dim)
        self.deep_linear1 = torch.nn.Linear(deep_dim, hidden_dim)
        self.deep_linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, wide_inputs, deep_inputs):
        wide_output = self.wide_linear(wide_inputs.float())
        deep_output = F.relu(self.deep_linear1(deep_inputs))
        deep_output = self.deep_linear2(deep_output)
        return wide_output + deep_output

# ============================================================================== #
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_data in dataloader:
        wide_inputs = batch_data["wide_inputs"].to(device)
        deep_inputs = batch_data["deep_inputs"].to(device)
        ratings = batch_data["rating"].to(device)
        optimizer.zero_grad()
        outputs = model(wide_inputs, deep_inputs)
        loss = criterion(outputs, ratings.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ============================================================================== #
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in dataloader:
            wide_inputs = batch_data["wide_inputs"].to(device)
            deep_inputs = batch_data["deep_inputs"].to(device)
            ratings = batch_data["rating"].to(device)
            outputs = model(wide_inputs, deep_inputs)
            loss = criterion(outputs, ratings.unsqueeze(1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ============================================================================== #
def monitor(wdb, ep, wdb_name, cw, cd, wd_shape=[33,4]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trndt, valdt = get_dataloader(cw, cd)
    model = WideAndDeep( wide_dim = wd_shape[0], 
                         deep_dim = wd_shape[1], 
                         hidden_dim = 64,
                         output_dim = 1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

    if wdb:
        from wandb import init
        monitor = init(project="employee", name=wdb_name, config = {"version":"v1.0"})
    for e in range(ep):
        train_loss = train(model, trndt, criterion, optimizer, device)
        test_loss = test(model, valdt, criterion, device)
        print(f'Epoch {e+1}/{ep} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
        if wdb: monitor.log({"Train_loss": train_loss, "Test_loss": test_loss})
    if wdb: monitor.finish()

# ============================================================================== #
if __name__ == '__main__':

    record = True
    epochs = 300

    monitor(wdb= record, ep=epochs, wdb_name='employee_all',
            cw = ['Eid','Interests','Languages', 'Projects_count','Skills'],
            cd = ['Firstname','Surname', 'Total_projects', 'Experience'],
            wd_shape=[33,4])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_eid',
            cw = ['Interests','Languages', 'Projects_count','Skills'],
            cd = ['Firstname','Surname', 'Total_projects', 'Experience'],
            wd_shape=[32,4])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_interests',
            cw = ['Eid','Languages', 'Projects_count','Skills'],
            cd = ['Firstname','Surname', 'Total_projects', 'Experience'],
            wd_shape=[30,4])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_languages',
            cw = ['Eid','Interests', 'Projects_count','Skills'],
            cd = ['Firstname','Surname', 'Total_projects', 'Experience'],
            wd_shape=[30,4])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_projects_count',
            cw = ['Eid','Interests','Languages','Skills'],
            cd = ['Firstname','Surname', 'Total_projects', 'Experience'],
            wd_shape=[27,4])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_skills',
            cw = ['Eid','Interests','Languages', 'Projects_count'],
            cd = ['Firstname','Surname', 'Total_projects', 'Experience'],
            wd_shape=[13,4])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_firstname',
            cw = ['Eid','Interests','Languages', 'Projects_count','Skills'],
            cd = ['Surname', 'Total_projects', 'Experience'],
            wd_shape=[33,3])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_surname',
            cw = ['Eid','Interests','Languages', 'Projects_count','Skills'],
            cd = ['Firstname', 'Total_projects', 'Experience'],
            wd_shape=[33,3])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_total_projects',
            cw = ['Eid','Interests','Languages', 'Projects_count','Skills'],
            cd = ['Firstname','Surname', 'Experience'],
            wd_shape=[33,3])

    monitor(wdb= record, ep=epochs, wdb_name='employee_no_experience',
            cw = ['Eid','Interests','Languages', 'Projects_count','Skills'],
            cd = ['Firstname','Surname', 'Total_projects'],
            wd_shape=[33,3])

# ============================================================================== #
