#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from mplsoccer import Pitch,Sbopen

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[3]:


parser= Sbopen()


# In[4]:


df, related, freeze, tactics = parser.event(3895250)


# In[5]:


df.columns


# In[6]:


df = df[df['team_name']=='Bayer Leverkusen']


# In[7]:


df['outcome_name'].unique()


# In[8]:


df['type_name'].unique()


# In[9]:


df['player_id'].unique()


# In[10]:


passes = df[df['type_name']=='Pass']
filtered = passes[['player_name','x','y','end_x', 'end_y','pass_recipient_name','outcome_id','outcome_name']]


# In[11]:


filtered[filtered['outcome_name'].isnull()]


# In[12]:


df_lineup = parser.lineup(3895250)
jersey_data=df_lineup[['player_id','jersey_number']]


# In[13]:


df= pd.merge(df,jersey_data,on='player_id')


# In[14]:


#player id is the jersey number
df['passer'] = df['jersey_number']
passes = df[df['type_name']=='Pass']
successful = passes[passes['outcome_name'].isnull()]


# In[15]:


rec= pd.to_numeric(successful['pass_recipient_id'],downcast='integer')
jersey_data.rename(columns={'player_id':'pass_recipient_id'},inplace=True)
jersey_data.rename(columns={'jersey_number':'pass_recipient'},inplace=True)
successful=pd.merge(df,jersey_data,on='pass_recipient_id')
successful


# In[16]:


subs = df[df['type_name']=='Substitution']
subs = subs['minute']
firstSub=subs.min()


# In[17]:


firstSub


# In[18]:


successful = successful[successful['minute']<firstSub]


# In[19]:


successful


# In[20]:


successful.dtypes


# In[21]:


successful


# In[22]:


df['pass_recipient_id'].unique()


# In[23]:


average_locations= successful.groupby('passer').agg({'x':['mean'],'y':['mean','count']})
average_locations.columns = ['x','y','count']


# In[24]:


average_locations


# In[25]:


pass_between = successful.groupby(['passer','pass_recipient']).id.count().reset_index()
pass_between.rename({'id':'pass_count'},axis='columns',inplace=True)
pass_between


# In[26]:


pass_between= pass_between.merge(average_locations,left_on='passer',right_index=True)
pass_between


# In[27]:


pass_between= pass_between.merge(average_locations,left_on='pass_recipient',right_index=True,suffixes=['','_end'])
pass_between


# In[28]:


pass_between =pass_between[pass_between['pass_count']>3]
pass_between.drop_duplicates()
pass_between


# In[29]:


average_locations['count'].values
pass_between['pass_count'].values


# In[30]:


pass_between


# In[31]:


pitch = Pitch(pitch_color='#FFFFFF', line_color='#B6BBC4')
# Set the figure size
fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)

# Set the face color of the figure
fig.set_facecolor('#FFFFFF')

# Draw arrows and nodes
# arrows = pitch.arrows(1.2 * pass_between.x, 0.8 * pass_between.y, 1.2 * pass_between.x_end, 0.8 * pass_between.y_end, ax=ax,color='red', alpha=0.4,width=3)
pass_lines = pitch.lines(1.2*pass_between.x, 0.8*pass_between.y, 1.2*pass_between.x_end, 0.8*pass_between.y_end, lw=pass_between.pass_count*0.5,color='red', zorder=1, ax=ax)
                 
nodes = pitch.scatter(1.2 * average_locations.x, 0.8 * average_locations.y, s=20*average_locations['count'].values, color='white', edgecolors='#a6aab3', linewidth=2, alpha=1, zorder=1, ax=ax)

                 
# Annotate average_locations
for index, row in average_locations.iterrows():
    pitch.annotate(index, xy=(1.2 * row.x, 0.8 * row.y), c='#161A30',fontweight='bold', va='center', ha='center', size=8, ax=ax)

# Add the endnote
ax.text(114, 85, 'Linkedin : @Oussama-BOUTROUFT', color='#0E2954', va='bottom', ha='center', fontsize=10)

# Add the title
ax.set_title('Passes NetWorkingMap Bayer LeverkuseN 2-1 FSV Mainz 05', color='red', va='center', ha='center', fontsize=12,fontweight='bold',pad=20,loc='center')
ax.annotate('Bundesliga 2023/24', xy=(0.5, 1), xytext=(0, 0),
             xycoords='axes fraction', textcoords='offset points',
             fontsize=10, color='#0E2954', va='top', ha='center')


plt.show()


# In[35]:


parser = Sbopen()
df, related, freeze, tactics = parser.event(3895250)
#get team names
team1, team2 = df.team_name.unique()
#A dataframe of shots
shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
team1, team2 = df.team_name.unique()
#A dataframe of shots
shots = df.loc[df['type_name'] == 'Shot'].set_index('id')


# In[37]:


pitch = Pitch(line_color = "black")
fig, ax = pitch.draw(figsize=(10, 7))
#Size of the pitch in yards (!!!)
pitchLengthX = 120
pitchWidthY = 80
#Plot the shots by looping through them.
for i,shot in shots.iterrows():
    #get the information
    x=shot['x']
    y=shot['y']
    goal=shot['outcome_name']=='Goal'
    team_name=shot['team_name']
    #set circlesize
    circleSize=2
    #plot England
    if (team_name==team1):
        if goal:
            shotCircle=plt.Circle((x,y),circleSize,color="red")
            plt.text(x+1,y-2,shot['player_name'])
        else:
            shotCircle=plt.Circle((x,y),circleSize,color="red")
            shotCircle.set_alpha(.2)
    #plot Sweden
    else:
        if goal:
            shotCircle=plt.Circle((pitchLengthX-x,pitchWidthY - y),circleSize,color="blue")
            plt.text(pitchLengthX-x+1,pitchWidthY - y - 2 ,shot['player_name'])
        else:
            shotCircle=plt.Circle((pitchLengthX-x,pitchWidthY - y),circleSize,color="blue")
            shotCircle.set_alpha(.2)
    ax.add_patch(shotCircle)
#set title
fig.suptitle("Passes NetWorkingMap Bayer LeverkuseN 2-1 FSV Mainz 05 shots", fontsize = 24)
fig.set_size_inches(10, 7)
plt.show()


# In[38]:


parser = Sbopen()
df, related, freeze, tactics = parser.event(3895250)
passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')


# In[42]:


#drawing pitch
pitch = Pitch(line_color = "black")
fig, ax = pitch.draw(figsize=(10, 7))

for i,thepass in passes.iterrows():
    #if pass made by Lucy Bronze
    if thepass['player_name']=='Granit Xhaka':
        x=thepass['x']
        y=thepass['y']
        #plot circle
        passCircle=plt.Circle((x,y),2,color="blue")
        passCircle.set_alpha(.2)
        ax.add_patch(passCircle)
        dx=thepass['end_x']-x
        dy=thepass['end_y']-y
        #plot arrow
        passArrow=plt.Arrow(x,y,dx,dy,width=3,color="red")
        ax.add_patch(passArrow)

ax.set_title("Granit Xhaka passes against Fc Mainz", fontsize = 24)
fig.set_size_inches(10, 7)
plt.show()


# In[45]:


mask_bronze = (df.type_name == 'Pass') & (df.player_name == "Granit Xhaka")
df_pass = df.loc[mask_bronze, ['x', 'y', 'end_x', 'end_y']]

pitch = Pitch(line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.arrows(df_pass.x, df_pass.y,
            df_pass.end_x, df_pass.end_y, color = "red", ax=ax['pitch'])
pitch.scatter(df_pass.x, df_pass.y, alpha = 0.2, s = 500, color = "red", ax=ax['pitch'])
fig.suptitle("Granit Xhaka Passes againt Fc Mainz", fontsize = 30)
plt.show()


# In[47]:


#prepare the dataframe of passes by England that were no-throw ins
mask_england = (df.type_name == 'Pass') & (df.team_name == "Bayer Leverkusen") & (df.sub_type_name != "Throw-in")
df_passes = df.loc[mask_england, ['x', 'y', 'end_x', 'end_y', 'player_name']]
#get the list of all players who made a pass
names = df_passes['player_name'].unique()

#draw 4x4 pitches
pitch = Pitch(line_color='black', pad_top=20)
fig, axs = pitch.grid(ncols = 4, nrows = 4, grid_height=0.85, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0.04, endnote_space=0.01)

#for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    #put player name over the plot
    ax.text(60, -10, name,
            ha='center', va='center', fontsize=14)
    #take only passes by this player
    player_df = df_passes.loc[df_passes["player_name"] == name]
    #scatter
    pitch.scatter(player_df.x, player_df.y, alpha = 0.2, s = 50, color = "red", ax=ax)
    #plot arrow
    pitch.arrows(player_df.x, player_df.y,
            player_df.end_x, player_df.end_y, color = "red", ax=ax, width=1)

#We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

#Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, 'Bayer Leverkusen passes against Sweden', ha='center', va='center', fontsize=30)
plt.show()


# In[ ]:




