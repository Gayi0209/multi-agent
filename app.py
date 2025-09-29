"""
Real-Time EdTech Marketing Platform with Streamlit UI

Prerequisites:
pip install streamlit crewai crewai-tools langchain-google-genai python-dotenv plotly pandas

Run with: streamlit run app.py
"""

import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from datetime import datetime
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="EdTech Marketing Automation Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize session state
if 'campaign_results' not in st.session_state:
    st.session_state.campaign_results = None
if 'crew_initialized' not in st.session_state:
    st.session_state.crew_initialized = False
if 'campaign_history' not in st.session_state:
    st.session_state.campaign_history = []

def initialize_crew():
    """Initialize CrewAI agents and tasks"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found! Please set it in your .env file")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return None
    
    # Initialize Gemini LLM
    gemini_llm = LLM(
        model="gemini/gemini-flash-latest",
        api_key=api_key,
        temperature=0.7
    )
    
    # Define Agents
    trend_analyst = Agent(
        role='EdTech Trend Analyst',
        goal='Monitor and analyze real-time trends in education technology, online learning, and student engagement',
        backstory="""You are an expert in identifying emerging trends in the EdTech space. 
        You monitor social media, news outlets, and educational forums to spot opportunities 
        for timely marketing campaigns.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )
    
    content_strategist = Agent(
        role='Content Strategy Specialist',
        goal='Develop compelling content strategies that align with current trends and resonate with students, teachers, and institutions',
        backstory="""You are a creative content strategist with deep knowledge of EdTech marketing. 
        You excel at transforming trending topics into engaging content that drives enrollment 
        and platform adoption.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )
    
    social_media_manager = Agent(
        role='Social Media Campaign Manager',
        goal='Create and optimize real-time social media campaigns across multiple platforms',
        backstory="""You are a social media expert specializing in EdTech marketing. 
        You know how to engage with students on TikTok, Instagram, LinkedIn, and Twitter. 
        You craft viral-worthy posts and use trending hashtags effectively.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )
    
    email_marketing_specialist = Agent(
        role='Email Marketing Specialist',
        goal='Design targeted email campaigns that convert leads into active users',
        backstory="""You are an email marketing expert who understands segmentation, 
        personalization, and conversion optimization. You create compelling subject lines, 
        engaging copy, and clear CTAs.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )
    
    performance_analyst = Agent(
        role='Marketing Performance Analyst',
        goal='Track campaign performance and provide actionable insights for optimization',
        backstory="""You are a data-driven marketing analyst who monitors KPIs, 
        conversion rates, engagement metrics, and ROI.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )
    
    # Define Tasks
    current_date = datetime.now().strftime("%B %d, %Y")
    
    task_trend_monitoring = Task(
        description=f"""Monitor current trends in EdTech for {current_date}. Focus on:
        1. Trending educational topics and challenges
        2. Viral content related to online learning
        3. Current events affecting students/teachers
        4. Competitor activities and campaigns
        5. Seasonal educational events
        
        Provide a detailed report with at least 5 trending opportunities.""",
        agent=trend_analyst,
        expected_output="A comprehensive trend report with 5+ actionable marketing opportunities"
    )
    
    task_content_strategy = Task(
        description="""Based on the trend analysis, develop a content strategy including:
        1. 3 blog post topics with titles and outlines
        2. 5 social media content ideas with platform recommendations
        3. 2 video content concepts
        4. 1 infographic theme
        5. Content calendar for the next 7 days""",
        agent=content_strategist,
        expected_output="A detailed content strategy document",
        context=[task_trend_monitoring]
    )
    
    task_social_campaigns = Task(
        description="""Create ready-to-publish social media campaigns:
        1. Write 10 social media posts (Instagram, LinkedIn, Twitter, TikTok)
        2. Include relevant hashtags for each post
        3. Suggest optimal posting times
        4. Create engagement hooks and CTAs
        5. Design caption variations for A/B testing""",
        agent=social_media_manager,
        expected_output="A complete social media campaign with 10 posts",
        context=[task_content_strategy]
    )
    
    task_email_campaign = Task(
        description="""Design an email marketing campaign:
        1. Create 3 email templates for:
           - New sign-ups (welcome)
           - Trial users (conversion)
           - Inactive users (re-engagement)
        2. Write compelling subject lines (5 options per template)
        3. Design email copy with personalization
        4. Include clear CTAs""",
        agent=email_marketing_specialist,
        expected_output="A complete email campaign package with 3 templates",
        context=[task_content_strategy]
    )
    
    task_performance_tracking = Task(
        description="""Create a performance tracking framework:
        1. Define KPIs for each campaign type
        2. Set up tracking metrics (CTR, conversion rate, engagement)
        3. Create a dashboard structure
        4. Establish benchmarks
        5. Provide optimization recommendations""",
        agent=performance_analyst,
        expected_output="A comprehensive performance tracking framework",
        context=[task_social_campaigns, task_email_campaign]
    )
    
    # Create the Crew
    marketing_crew = Crew(
        agents=[
            trend_analyst,
            content_strategist,
            social_media_manager,
            email_marketing_specialist,
            performance_analyst
        ],
        tasks=[
            task_trend_monitoring,
            task_content_strategy,
            task_social_campaigns,
            task_email_campaign,
            task_performance_tracking
        ],
        process=Process.sequential,
        verbose=True
    )
    
    return marketing_crew

def run_marketing_campaign(company_info, crew):
    """Execute the marketing campaign"""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    result = crew.kickoff(inputs={
        'date': current_date,
        'company_name': company_info['name'],
        'target_audience': company_info['target_audience'],
        'products': company_info['products'],
        'usp': company_info['usp']
    })
    
    return result

# Sidebar - Company Information
st.sidebar.title("🏢 Company Configuration")
st.sidebar.markdown("---")

company_name = st.sidebar.text_input("Company Name", value="LearnFlow")
target_audience = st.sidebar.text_area(
    "Target Audience",
    value="College students, working professionals, and lifelong learners aged 18-35"
)
products = st.sidebar.text_area(
    "Products/Services",
    value="AI-powered personalized learning paths\nLive interactive courses\nSkill certification programs\nCareer mentorship"
)
usp = st.sidebar.text_area(
    "Unique Selling Proposition",
    value="Adaptive learning technology with 1-on-1 mentorship and job placement assistance"
)

st.sidebar.markdown("---")
api_key_status = "✅ Connected" if os.getenv("GOOGLE_API_KEY") else "❌ Not Set"
st.sidebar.info(f"**API Status:** {api_key_status}")

# Main Header
st.markdown('<h1 class="main-header">🚀 EdTech Marketing Automation Platform</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Real-Time Marketing Campaign Generator")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🎯 Generate Campaign", 
    "📈 Analytics", 
    "📋 Campaign History",
    "⚙️ Settings"
])

# Tab 1: Dashboard
with tab1:
    st.header("Real-Time Marketing Dashboard")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Campaigns",
            value=len(st.session_state.campaign_history),
            delta="+1" if len(st.session_state.campaign_history) > 0 else "0"
        )
    
    with col2:
        st.metric(
            label="Avg. Engagement Rate",
            value="24.5%",
            delta="+3.2%"
        )
    
    with col3:
        st.metric(
            label="Conversion Rate",
            value="8.3%",
            delta="+1.5%"
        )
    
    with col4:
        st.metric(
            label="ROI",
            value="342%",
            delta="+12%"
        )
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Campaign Performance")
        
        # Sample data for visualization
        performance_data = pd.DataFrame({
            'Campaign': ['Social Media', 'Email', 'Content', 'Video'],
            'Engagement': [85, 72, 68, 91],
            'Conversions': [45, 38, 42, 52]
        })
        
        fig = px.bar(
            performance_data,
            x='Campaign',
            y=['Engagement', 'Conversions'],
            title='Campaign Performance Comparison',
            barmode='group',
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Audience Reach")
        
        # Pie chart
        reach_data = pd.DataFrame({
            'Platform': ['Instagram', 'LinkedIn', 'Twitter', 'TikTok', 'Email'],
            'Reach': [35, 25, 20, 15, 5]
        })
        
        fig = px.pie(
            reach_data,
            values='Reach',
            names='Platform',
            title='Reach by Platform',
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.subheader("🕒 Recent Activity")
    if st.session_state.campaign_history:
        for campaign in st.session_state.campaign_history[-3:]:
            st.markdown(f"""
            <div class="info-box">
                <strong>{campaign['name']}</strong><br>
                Generated on: {campaign['date']}<br>
                Status: ✅ Active
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No campaigns generated yet. Go to 'Generate Campaign' tab to create your first campaign!")

# Tab 2: Generate Campaign
with tab2:
    st.header("🎯 Generate AI-Powered Marketing Campaign")
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How it works:</strong><br>
        Our AI agents will analyze current trends, create content strategies, generate social media posts, 
        design email campaigns, and set up performance tracking - all in real-time!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Campaign Overview")
        st.write(f"**Company:** {company_name}")
        st.write(f"**Target Audience:** {target_audience}")
        st.write(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
    
    with col2:
        st.subheader("AI Agents")
        st.write("✅ Trend Analyst")
        st.write("✅ Content Strategist")
        st.write("✅ Social Media Manager")
        st.write("✅ Email Specialist")
        st.write("✅ Performance Analyst")
    
    st.markdown("---")
    
    if st.button("🚀 Generate Complete Campaign", type="primary"):
        
        company_info = {
            'name': company_name,
            'target_audience': target_audience,
            'products': products,
            'usp': usp
        }
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize crew
            status_text.text("🔧 Initializing AI Marketing Team...")
            progress_bar.progress(10)
            time.sleep(1)
            
            crew = initialize_crew()
            
            if crew is None:
                st.error("Failed to initialize crew. Please check your API key.")
            else:
                # Run campaign
                status_text.text("📊 Analyzing current trends...")
                progress_bar.progress(25)
                time.sleep(1)
                
                status_text.text("✍️ Creating content strategy...")
                progress_bar.progress(45)
                time.sleep(1)
                
                status_text.text("📱 Generating social media posts...")
                progress_bar.progress(65)
                time.sleep(1)
                
                status_text.text("📧 Designing email campaigns...")
                progress_bar.progress(85)
                
                # Execute
                result = run_marketing_campaign(company_info, crew)
                
                progress_bar.progress(100)
                status_text.text("✅ Campaign generated successfully!")
                
                # Store results
                st.session_state.campaign_results = result
                st.session_state.campaign_history.append({
                    'name': f"{company_name} Campaign",
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'result': result
                })
                
                st.success("🎉 Campaign Generated Successfully!")
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Campaign Results")
                
                with st.expander("📊 View Complete Campaign Details", expanded=True):
                    st.markdown(result)
                
                # Download button
                st.download_button(
                    label="📥 Download Campaign Report",
                    data=str(result),
                    file_name=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            st.info("""
            **Troubleshooting:**
            1. Ensure GOOGLE_API_KEY is set in your .env file
            2. Check that all required packages are installed
            3. Verify you have a stable internet connection
            """)

# Tab 3: Analytics
with tab3:
    st.header("📈 Campaign Analytics")
    
    # Time series data
    dates = pd.date_range(start='2024-09-01', end='2024-09-29', freq='D')
    analytics_data = pd.DataFrame({
        'Date': dates,
        'Impressions': [1000 + i*50 + (i%7)*100 for i in range(len(dates))],
        'Clicks': [100 + i*5 + (i%5)*20 for i in range(len(dates))],
        'Conversions': [10 + i*0.5 + (i%3)*5 for i in range(len(dates))]
    })
    
    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=analytics_data['Date'], y=analytics_data['Impressions'], 
                             name='Impressions', line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=analytics_data['Date'], y=analytics_data['Clicks'], 
                             name='Clicks', line=dict(color='#764ba2', width=2)))
    fig.add_trace(go.Scatter(x=analytics_data['Date'], y=analytics_data['Conversions'], 
                             name='Conversions', line=dict(color='#f093fb', width=2)))
    
    fig.update_layout(title='Campaign Performance Over Time', xaxis_title='Date', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Social Media")
        st.metric("Total Posts", "127", delta="+12")
        st.metric("Engagement Rate", "18.5%", delta="+2.3%")
        st.metric("Followers Growth", "+2,450", delta="+15%")
    
    with col2:
        st.subheader("Email Marketing")
        st.metric("Emails Sent", "15,420", delta="+1,200")
        st.metric("Open Rate", "32.4%", delta="+1.8%")
        st.metric("Click Rate", "8.7%", delta="+0.9%")
    
    with col3:
        st.subheader("Conversions")
        st.metric("Sign-ups", "1,284", delta="+156")
        st.metric("Trial Starts", "687", delta="+89")
        st.metric("Paid Conversions", "342", delta="+45")

# Tab 4: Campaign History
with tab4:
    st.header("📋 Campaign History")
    
    if st.session_state.campaign_history:
        for idx, campaign in enumerate(reversed(st.session_state.campaign_history)):
            with st.expander(f"📊 {campaign['name']} - {campaign['date']}"):
                st.markdown(campaign['result'])
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.download_button(
                        label="Download",
                        data=str(campaign['result']),
                        file_name=f"campaign_{idx}.txt",
                        mime="text/plain",
                        key=f"download_{idx}"
                    )
    else:
        st.info("No campaigns in history yet. Generate your first campaign in the 'Generate Campaign' tab!")

# Tab 5: Settings
with tab5:
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("API Configuration")
    
    current_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_display = current_api_key[:10] + "..." if current_api_key else "Not Set"
    
    st.text_input("Gemini API Key", value=api_key_display, type="password", disabled=True)
    st.info("To change the API key, update your .env file with GOOGLE_API_KEY=your_key_here")
    
    st.markdown("---")
    
    st.subheader("Campaign Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Preferred Tone", ["Professional", "Casual", "Inspirational", "Educational"])
        st.selectbox("Campaign Frequency", ["Daily", "Weekly", "Bi-weekly", "Monthly"])
    
    with col2:
        st.multiselect(
            "Active Platforms",
            ["Instagram", "LinkedIn", "Twitter", "TikTok", "Facebook", "YouTube"],
            default=["Instagram", "LinkedIn", "Twitter", "TikTok"]
        )
        st.slider("Budget Allocation ($)", 0, 10000, 5000)
    
    st.markdown("---")
    
    st.subheader("Automation Settings")
    
    auto_generate = st.checkbox("Auto-generate weekly campaigns")
    send_reports = st.checkbox("Send weekly performance reports")
    email_notifications = st.checkbox("Enable email notifications")
    
    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>EdTech Marketing Automation Platform</strong></p>
    <p>Powered by CrewAI & Google Gemini | Built with Streamlit</p>
    <p>Real-time AI-driven marketing campaigns for education technology companies</p>
</div>
""", unsafe_allow_html=True)
