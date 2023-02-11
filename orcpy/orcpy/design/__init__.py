import pandas as pd
import numpy as np
import CoolProp.CoolProp as cp
import CoolProp
from CoolProp.CoolProp import PropsSI
import numpy as np
import sympy
import copy
import pandas as pd
import pina as pin
from pina import PinchAnalyzer, make_stream
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import json
import math
from math import pi , radians, acos, cos, sin, sqrt
from math import log as ln
bar = 101325

class ORC:
  
  def working_fluids(fluid_list):

    fluid = ["R218","R1234yf","R227EA","RC318","R236FA","R124","R236EA","isobutane","R245fa","ipentane","R123","Pentane","Cyclopentane","Hexane","Benzene"]
    if fluid_list == "all":
      return(fluid)

    else:
      fluids = []
      for i in range(len(fluid_list)):
        try:
          fluids = np.append(fluids,fluid[fluid.index(fluid_list[i])])
        except ValueError:
            print(str("Please select working fluids from the list \n <<")+str(fluid_list[i])+str(">> is not in the list") )
      return(fluids)
    
  def w_fluid(fluid_list):

    fluid = ["R218","R1234yf","R227EA","RC318","R236FA","R124","R236EA","isobutane","R245fa","ipentane","R123","Pentane","Cyclopentane","Hexane","Benzene"]
    if fluid_list == "all":
      return(fluid)

    else:
      fluids = []
      for i in range(len(fluid_list)):
        try:
          fluids = np.append(fluids,fluid[fluid.index(fluid_list[i])])
        except ValueError:
            print(str("Please select working fluids from the list \n <<")+str(fluid_list[i])+str(">> is not in the list") )
      return(fluids)

  def information():
    bar = 101325  #[Pa]
    T1 = float(input("Please enter the inlet temperature for waste stream in C: "))
    T2 = float(input("Please enter the minimum allowable temperature for waste stream in C: "))
    if T2 >= T1:
        while T2 >= T1:
          print("initial temperature must be larger than minimum allowable temperature")
          T1 = float(input("Please enter the inlet temperature for waste stream in C: "))
          T2 = float(input("Please enter the minimum allowable temperature for waste stream in C: "))


    turbine_eff = float(input("Please enter the turbine efficiency [%]: "))
    pump_eff = float(input("Please enter the pump efficiency [%]: ")) 
    if turbine_eff > 100 or pump_eff > 100:
      while turbine_eff > 100 or pump_eff > 100:
        print("Efficiencies must be less than 100")
        turbine_eff = float(input("Please enter the turbine efficiency [%]: "))
        pump_eff = float(input("Please enter the pump efficiency [%]: "))   
      
    P = float(input("Please enter the power of waste stream [kW]: "))
    C = float(input("Please enter the heat capacity of waste stream [kJ/kgC]: "))
    min_tem_diff = float(input("Please enter the minimum temperature difference in heat exchangers [C]: "))
    min_cond_temp = float(input("Please enter the minimum temperature in condenser [C]: "))
    return(T1, T2, turbine_eff/100, pump_eff/100, P, C, bar, min_tem_diff, min_cond_temp)
  

  def info(Tw1, Tw2, turbine_eff, pump_eff, fluids):

    if Tw2 >= Tw1:
        return(False)
    if turbine_eff > 100 or pump_eff > 100:
      return(False)
    if fluids == []:
      return(False)
    else:
        return(True)
    

  def pressure_levels(working_fluid,bar,min_cond_temp):
    TIP_limit = cp.PropsSI("Pcrit",working_fluid) * 0.85 #[Pa]
    TOP = max( bar , cp.PropsSI('P','T',min_cond_temp+273,'Q',1,working_fluid)) #[Pa]
    TIP_levels = np.arange(TOP + bar, TIP_limit, bar)
    return(TIP_limit,TIP_levels,TOP)

  def Condenser(working_fluid,TOP):
    P1 = copy.copy(TOP) #[Pa]
    S1 = cp.PropsSI('S','P',P1,'Q',0,working_fluid) #[J/kg]
    H1 = cp.PropsSI('H','P',P1,'Q',0,working_fluid) #[J/kg]
    T1 = cp.PropsSI('T','P',P1,'Q',0,working_fluid) #[K]
    return(P1, S1, H1, T1)

  def Pump(working_fluid, TIP_level, pump_eff, S1, H1):
    P2s = copy.copy(TIP_level) #[Pa]
    S2s = copy.copy(S1) #[J/kg]
    H2s = cp.PropsSI('H','P',P2s,'S',S2s,working_fluid) #[J/kg]
    T2s = cp.PropsSI('T','P',P2s,'S',S2s,working_fluid) #[K]

    H2 = H1 + (H2s-H1)/pump_eff #[J/kg]
    P2 = copy.copy(TIP_level) #[Pa]
    S2 = cp.PropsSI('S','P',P2,'H',H2,working_fluid) #[J/kg]
    T2 = cp.PropsSI('T','P',P2,'H',H2,working_fluid) #[K]
    return(P2, S2, H2, T2)

  def Evaporator(working_fluid,TIP_level):
    P3 = copy.copy(TIP_level) #[Pa]
    S3 = cp.PropsSI('S','P',P3,'Q',0,working_fluid) #[J/kg]
    H3 = cp.PropsSI('H','P',P3,'Q',0,working_fluid) #[J/kg]
    T3 = cp.PropsSI('T','P',P3,'Q',0,working_fluid) #[K]

    P4 = copy.copy(TIP_level) #[Pa]
    T4 = cp.PropsSI('T','P',P4,'Q',1,working_fluid) #[K]
    H4 = cp.PropsSI('H','P',P4,'Q',1,working_fluid) #[J/kg]
    S4 = cp.PropsSI('S','P',P4,'Q',1,working_fluid) #[J/kg]
    return(P3, S3, H3, T3, P4, S4, H4, T4)

  def Turbine(working_fluid, TOP, turbine_eff, S4, H4):
    S5s = copy.copy(S4) #[J/kg]
    P5s = copy.copy(TOP) #[Pa]
    H5s = cp.PropsSI('H','P',P5s,'S',S5s,working_fluid) #[J/kg]
    T5s = cp.PropsSI('T','P',P5s,'S',S5s,working_fluid) #[K]

    H5 = H4+(H5s-H4)*turbine_eff
    P5 = copy.copy(TOP) #[Pa]
    S5 = cp.PropsSI('S','P',P5,'H',H5,working_fluid) #[J/kg]
    T5 = cp.PropsSI('T','P',P5,'H',H5,working_fluid) #[K]
    return(P5, S5, H5, T5)

  def closing_cycle(P5, S5, H5, T5,working_fluid):
    P6s = copy.copy(P5) #[Pa]
    S6s = cp.PropsSI('S','P',P5,'Q',1,working_fluid) #[J/kg]
    H6s = cp.PropsSI('H','P',P6s,'Q',1,working_fluid) #[J/kg]
    T6s = cp.PropsSI('T','P',P6s,'Q',1,working_fluid) #[K]

    if S6s < S5:
      P6 = copy.copy(P6s)
      S6 = copy.copy(S6s)
      H6 = copy.copy(H6s)
      T6 = copy.copy(T6s)
    else:
      P6 = copy.copy(P5)
      S6 = copy.copy(S5) 
      H6 = copy.copy(H5)
      T6 = copy.copy(T5)
    return(P6, S6, H6, T6)
    
  def pinch(C, P, Tw1, Tw2, T2, T3, T4, H2, H3, H4, min_tem_diff):
    M = P / ((Tw1-Tw2)*C)
    m_wf = M * 5
    pinch_analysis = True

    while pinch_analysis == True:

      Q_pre_heater = m_wf*(H2-H3)/1000 #[kW]
      Q_evaporator = m_wf*(H3-H4)/1000 #[kW]

      cold_1 = make_stream(Q_pre_heater,T2-273,T3-273)
      cold_2 = make_stream(Q_evaporator,T3-273,T4-273)
      hot_1 = make_stream(P,Tw1,Tw2)

      analyzer = PinchAnalyzer((min_tem_diff / 2))
      analyzer.add_streams(cold_1, hot_1, cold_2)

      if analyzer.hot_utility_target == 0:
        pinch_analysis = False

      elif m_wf <= 0:
        pinch_analysis = False

      else:
        m_wf -= 0.1
      
    return(m_wf, Q_pre_heater, Q_evaporator)

  def result(m_wf, working_fluid, bar,
             H1, H2, H3, H4, H5, H6,
             T1, T2, T3, T4, T5, T6,
             S1, S2, S3, S4, S5, S6,
             TIP_level, TOP, opt,
             Q_pre_heater, Q_evaporator):
    if m_wf > 0 :

      W_turbine = (m_wf*(H4-H5)/1000) #[kW]
      W_pump = (m_wf*(H2-H1)/1000) #[kW]
      W_net = W_turbine - W_pump
      ETA = W_net/abs(Q_pre_heater+Q_evaporator)

    else:
      W_net = 0
      ETA = 0

    if W_net > opt.at[working_fluid,'Net power output']:

      opt.at[working_fluid,'Cycle efficiency'] = ETA*100 #[%]
      opt.at[working_fluid,'Net power output'] = W_net #[kW]     
      opt.at[working_fluid,'TIP'] = TIP_level/bar #[bar]     
      opt.at[working_fluid,'TOP'] = TOP/bar #[bar]
      opt.at[working_fluid,'flow rate'] = m_wf #[kg/sec]
      opt.at[working_fluid,'Recovered Heat'] = -(Q_pre_heater)-(Q_evaporator) #[kw]
      opt.at[working_fluid,'Rejected Heat'] = (m_wf*(H5-H6)/1000) + (m_wf*(H6-H1)/1000) #[kW]
      opt.at[working_fluid,'Turbine Work'] = W_turbine #[kw]
      opt.at[working_fluid,'Pump Work'] = W_pump #[kw]
      opt.at[working_fluid,'S1'] = S1/1000 #[kJ/kg]
      opt.at[working_fluid,'S2'] = S2/1000 #[kJ/kg]   
      opt.at[working_fluid,'S3'] = S3/1000 #[kJ/kg]
      opt.at[working_fluid,'S4'] = S4/1000 #[kJ/kg]
      opt.at[working_fluid,'S5'] = S5/1000 #[kJ/kg]
      opt.at[working_fluid,'S6'] = S6/1000 #[kJ/kg]
      opt.at[working_fluid,'T1'] = T1-273 #[c]      
      opt.at[working_fluid,'T2'] = T2-273 #[c]  
      opt.at[working_fluid,'T3'] = T3-273 #[c]   
      opt.at[working_fluid,'T4'] = T4-273 #[c]       
      opt.at[working_fluid,'T5'] = T5-273 #[c]      
      opt.at[working_fluid,'T6'] = T6-273 #[c]  
      opt.at[working_fluid,'H1'] = H1/1000 #[kJ/kg]  
      opt.at[working_fluid,'H2'] = H2/1000 #[kJ/kg]
      opt.at[working_fluid,'H3'] = H3/1000 #[kJ/kg] 
      opt.at[working_fluid,'H4'] = H4/1000 #[kJ/kg]
      opt.at[working_fluid,'H5'] = H5/1000 #[kJ/kg]
      opt.at[working_fluid,'H6'] = H6/1000 #[kJ/kg]
    return (opt)

  def Visualization(opt):
    suitble_wf = opt.loc[opt['Net power output'].idxmax()]
    fluid = suitble_wf.name

    Tc = cp.PropsSI("Tcrit",fluid)
    tt =int(Tc-280)

    x = np.zeros((tt,3))
    y = np.zeros((2*tt,2))

    for i in range(tt):
      x[i,0] = Tc-i-273
      x[i,1] = cp.PropsSI('S','T',Tc-i,'Q',0,fluid)/1000
      x[i,2] = cp.PropsSI('S','T',Tc-i,'Q',1,fluid)/1000

    y[0:tt,0] = x[:,0]
    y[tt:2*tt,0] = np.flip(x[:,0])
    y[0:tt,1] = x[:,1]
    y[tt:2*tt,1] = np.flip(x[:,2])



    fig = go.Figure(go.Scatter(
        x=y[:,1],
        y=y[:,0],
          fill="toself",
          fillcolor="LightSkyBlue",
          name = 'Phase diagram'
    ))
    fig.add_trace(go.Scatter(
        x=[suitble_wf['S1'],
           suitble_wf['S2'],
           suitble_wf['S3'],
           suitble_wf['S4'],
           suitble_wf['S5'],
           suitble_wf['S6'],
           suitble_wf['S1']],
           y=[suitble_wf['T1'],
              suitble_wf['T2'],
              suitble_wf['T3'],
              suitble_wf['T4'],
              suitble_wf['T5'],
              suitble_wf['T6'],
              suitble_wf['T1']],
              name = 'Cycle',
              text = ["", str(int(suitble_wf[6]))+" bar", 
                      str(int(suitble_wf[6]))+" bar",
                      str(int(suitble_wf[6]))+" bar",
                      str(int(suitble_wf[7]))+" bar",
                      str(int(suitble_wf[7]))+" bar"],
                      mode="lines+markers+text",
                      textposition="top right",
                      textfont=dict(size=15,color="black"),
                      marker=dict(size=10),
                      line=dict(width=3)))

    fig.update_layout(
        title={'text': f'{fluid} T-S diagram ',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title="Entropy [kJ/kg k]",
        yaxis_title="Temperature [\u2103]",
        font=dict(
            size=14,
            color="BLACK"
        )
    )
    fig.update_layout(width=1200,height=800)
    fig.update_layout(font_size=14)

    return (fig)

  def cycle(fluid_list):

    '''
    selecting working fluids
    '''
    fluids = ORC.working_fluids(fluid_list)

    '''
    Input variables
    '''
    Tw1, Tw2, turbine_eff, pump_eff, P, C, bar, min_tem_diff, min_cond_temp = ORC.information()

    '''
    result frame
    '''
  
    opt = pd.DataFrame(np.zeros((len(fluids), 6)),index = fluids,
    columns=["Cycle efficiency","Net power output",'Turbine Work', 'Pump Work',"Recovered Heat","Rejected Heat"])

    '''
    designing cylces
    '''

    for working_fluid in fluids:
      '''
      pressure levels
      '''
      TIP_limit,TIP_levels,TOP = ORC.pressure_levels(working_fluid,bar,min_cond_temp)

      for TIP_level in TIP_levels:

        '''
        Condenser
        '''
        P1, S1, H1, T1 = ORC.Condenser(working_fluid, TOP)
        '''
        Pump  
        '''
        P2, S2, H2, T2 = ORC.Pump(working_fluid, TIP_level, pump_eff, S1, H1)
        '''
        Evaporator and pre heater
        '''
        P3, S3, H3, T3, P4, S4, H4, T4 =  ORC.Evaporator(working_fluid,TIP_level)
        '''
        Turbine
        '''
        P5, S5, H5, T5 = ORC.Turbine(working_fluid, TOP, turbine_eff, S4, H4)
        '''
        Closing the cycle
        '''
        P6, S6, H6, T6 = ORC.closing_cycle(P5, S5, H5, T5,working_fluid)

        '''
        mass flow rate
        '''
        m_wf, Q_pre_heater, Q_evaporator = ORC.pinch(C, P, Tw1, Tw2, T2, T3, T4, H2, H3, H4, min_tem_diff)      

        '''
        results
        '''
        opt = ORC.result(m_wf, working_fluid, bar,
              H1, H2, H3, H4, H5, H6,
              T1, T2, T3, T4, T5, T6,
              S1, S2, S3, S4, S5, S6,
              TIP_level, TOP, opt,
              Q_pre_heater, Q_evaporator)
        '''
        Visualization
        '''
        fig = ORC.Visualization(opt)

    return(opt,fig)

  def model(Tw1, Tw2, P, C
            , turbine_eff, pump_eff
            , min_tem_diff, min_cond_temp
            , fluid_list):

    '''
    selecting working fluids
    '''
    fluids = ORC.w_fluid(fluid_list)

    '''
    result frame
    '''
  
    opt = pd.DataFrame(np.zeros((len(fluids), 6)),index = fluids,
    columns=["Cycle efficiency","Net power output",'Turbine Work', 'Pump Work',"Recovered Heat","Rejected Heat"])

    '''
    Input variables
    '''
    if ORC.info(Tw1, Tw2, turbine_eff, pump_eff, fluids) == True:
      turbine_eff = turbine_eff/100
      pump_eff = pump_eff/100


      '''
      designing cylces
      '''

      for working_fluid in fluids:
        '''
        pressure levels
        '''
        TIP_limit,TIP_levels,TOP = ORC.pressure_levels(working_fluid,bar,min_cond_temp)

        for TIP_level in TIP_levels:

          '''
          Condenser
          '''
          P1, S1, H1, T1 = ORC.Condenser(working_fluid, TOP)
          '''
          Pump  
          '''
          P2, S2, H2, T2 = ORC.Pump(working_fluid, TIP_level, pump_eff, S1, H1)
          '''
          Evaporator and pre heater
          '''
          P3, S3, H3, T3, P4, S4, H4, T4 =  ORC.Evaporator(working_fluid,TIP_level)
          '''
          Turbine
          '''
          P5, S5, H5, T5 = ORC.Turbine(working_fluid, TOP, turbine_eff, S4, H4)
          '''
          Closing the cycle
          '''
          P6, S6, H6, T6 = ORC.closing_cycle(P5, S5, H5, T5,working_fluid)

          '''
          mass flow rate
          '''
          m_wf, Q_pre_heater, Q_evaporator = ORC.pinch(C, P, Tw1, Tw2, T2, T3, T4, H2, H3, H4, min_tem_diff)      

          '''
          results
          '''
          opt = ORC.result(m_wf, working_fluid, bar,
                H1, H2, H3, H4, H5, H6,
                T1, T2, T3, T4, T5, T6,
                S1, S2, S3, S4, S5, S6,
                TIP_level, TOP, opt,
                Q_pre_heater, Q_evaporator)
          '''
          Visualization
          '''
          fig = ORC.Visualization(opt)
    else:
      opt = "input variables are not valid"
      fig = "input variables are not valid"

    return(opt,fig)
