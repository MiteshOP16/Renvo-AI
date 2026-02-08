"""
Database Connector Module for Renvo AI
Provides MySQL database connectivity for importing data
"""

import streamlit as st
import pandas as pd
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import urllib.parse


class MySQLConnector:
    """MySQL Database Connector for importing data into Renvo AI"""
    
    def __init__(self):
        self.engine = None
        self.connection_info = {}
        self.is_connected = False
    
    def connect(self, host: str, port: int, database: str, 
                username: str, password: str, use_ssl: bool = False) -> tuple[bool, str]:
        """
        Establish connection to MySQL database
        
        Args:
            host: Database host address
            port: Database port (default 3306)
            database: Database name
            username: Database username
            password: Database password
            use_ssl: Whether to use SSL connection
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # URL encode password to handle special characters
            encoded_password = urllib.parse.quote_plus(password)
            
            # Build connection string
            ssl_args = "?ssl_disabled=false" if use_ssl else ""
            connection_string = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}{ssl_args}"
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False           # Disable SQL logging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Store connection info
            self.connection_info = {
                'host': host,
                'port': port,
                'database': database,
                'username': username
            }
            self.is_connected = True
            
            return True, f"Successfully connected to {database}@{host}:{port}"
            
        except SQLAlchemyError as e:
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            self.is_connected = False
            return False, f"Connection failed: {error_msg}"
        except Exception as e:
            self.is_connected = False
            return False, f"Connection failed: {str(e)}"
    
    def get_tables(self) -> tuple[List[str], str]:
        """
        Get list of all tables in the connected database
        
        Returns:
            Tuple of (table_list: List[str], message: str)
        """
        if not self.is_connected or not self.engine:
            return [], "Not connected to database"
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            return tables, f"Found {len(tables)} tables"
        except Exception as e:
            return [], f"Failed to fetch tables: {str(e)}"
    
    def get_table_info(self, table_name: str) -> tuple[pd.DataFrame, str]:
        """
        Get table schema/structure information
        
        Args:
            table_name: Name of the table
            
        Returns:
            Tuple of (schema_df: DataFrame, message: str)
        """
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            query = f"DESCRIBE `{table_name}`"
            df = pd.read_sql(query, self.engine)
            return df, f"Schema for table '{table_name}'"
        except Exception as e:
            return pd.DataFrame(), f"Failed to get table info: {str(e)}"
    
    def get_row_count(self, table_name: str) -> tuple[int, str]:
        """
        Get the number of rows in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Tuple of (count: int, message: str)
        """
        if not self.is_connected or not self.engine:
            return 0, "Not connected to database"
        
        try:
            query = f"SELECT COUNT(*) as count FROM `{table_name}`"
            df = pd.read_sql(query, self.engine)
            count = int(df['count'].iloc[0])
            return count, f"Table '{table_name}' has {count:,} rows"
        except Exception as e:
            return 0, f"Failed to count rows: {str(e)}"
    
    def preview_table(self, table_name: str, limit: int = 100) -> tuple[pd.DataFrame, str]:
        """
        Preview table data with limited rows
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            
        Returns:
            Tuple of (preview_df: DataFrame, message: str)
        """
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
            df = pd.read_sql(query, self.engine)
            return df, f"Showing {len(df)} rows from '{table_name}'"
        except Exception as e:
            return pd.DataFrame(), f"Failed to preview table: {str(e)}"
    
    def import_table(self, table_name: str, limit: Optional[int] = None) -> tuple[pd.DataFrame, str]:
        """
        Import entire table as DataFrame
        
        Args:
            table_name: Name of the table to import
            limit: Optional limit on number of rows
            
        Returns:
            Tuple of (data_df: DataFrame, message: str)
        """
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            if limit:
                query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
            else:
                query = f"SELECT * FROM `{table_name}`"
            
            df = pd.read_sql(query, self.engine)
            return df, f"Imported {len(df):,} rows and {len(df.columns)} columns from '{table_name}'"
        except Exception as e:
            return pd.DataFrame(), f"Failed to import table: {str(e)}"
    
    def import_query(self, query: str) -> tuple[pd.DataFrame, str]:
        """
        Import data using custom SQL query
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (data_df: DataFrame, message: str)
        """
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            df = pd.read_sql(query, self.engine)
            return df, f"Query returned {len(df):,} rows and {len(df.columns)} columns"
        except Exception as e:
            return pd.DataFrame(), f"Failed to execute query: {str(e)}"
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information (without password)"""
        return self.connection_info.copy()
    
    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.is_connected = False
        self.connection_info = {}
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()


class SupabaseConnector:
    """Supabase Database Connector for importing data into Renvo AI"""
    
    def __init__(self):
        self.engine = None
        self.connection_info = {}
        self.is_connected = False
    
    def connect(self, project_url: str, db_password: str, 
                custom_host: Optional[str] = None, 
                custom_port: int = 5432) -> tuple[bool, str]:
        """
        Establish connection to Supabase PostgreSQL database
        
        Args:
            project_url: Supabase project URL or project reference
            db_password: Database password
            custom_host: Optional override for database host
            custom_port: Optional override for database port
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            project_ref = ""
            if not custom_host:
                # Clean input
                project_url = project_url.strip()
                if project_url.startswith('https://'):
                    project_url = project_url[8:]
                elif project_url.startswith('http://'):
                    project_url = project_url[7:]
                
                if project_url.endswith('/'):
                    project_url = project_url[:-1]
                
                # Check if user entered a full host already (Direct or Pooler)
                if any(ext in project_url for ext in ['.supabase.co', '.supabase.com', '.pooler.supabase.com']):
                    host = project_url
                    project_ref = project_url.split('.')[0]
                    # If it's a pooler host, auto-set port to 6543
                    if '.pooler.' in project_url:
                        port = 6543
                    else:
                        port = 5432
                else:
                    # Treat as project reference
                    project_ref = project_url
                    host = f"db.{project_ref}.supabase.co"
                    port = 5432
            else:
                host = custom_host
                port = custom_port
                project_ref = project_url
            
            database = "postgres"
            username = "postgres"
            
            # URL encode password to handle special characters
            encoded_password = urllib.parse.quote_plus(db_password)
            
            # Build PostgreSQL connection string
            connection_string = f"postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}?sslmode=require"
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Store connection info
            self.connection_info = {
                'project_ref': project_ref,
                'host': host,
                'port': port,
                'database': database
            }
            self.is_connected = True
            
            return True, f"Successfully connected to Supabase project: {project_ref}"
            
        except SQLAlchemyError as e:
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            self.is_connected = False
            
            # AUTOMATIC RETRY: If direct connection fails on 5432, try the pooler on 6543
            # This is common for users on IPv4-only networks
            if not custom_host and port == 5432 and ("Cannot assign requested address" in error_msg or "Network is unreachable" in error_msg):
                try:
                    # Supabase pooler host usually uses a different subdomain pattern:
                    # Direct: db.ref.supabase.co (5432)
                    # Pooler: aws-0-ref.pooler.supabase.com (6543)
                    # However, some projects use the same host but different port. 
                    # We'll first try the same host with 6543 which is often supported.
                    pooler_port = 6543
                    pooler_conn_string = f"postgresql+psycopg2://{username}:{encoded_password}@{host}:{pooler_port}/{database}?sslmode=require"
                    
                    self.engine = create_engine(pooler_conn_string, pool_pre_ping=True, pool_recycle=3600, echo=False)
                    
                    with self.engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    
                    self.connection_info = {'project_ref': project_ref, 'host': host, 'port': pooler_port, 'database': database}
                    self.is_connected = True
                    return True, f"Successfully connected to Supabase project: {project_ref} (via Connection Pooler)"
                except Exception:
                    # If auto-retry fails, proceed to detailed error message below
                    pass

            # Specific handling for IPv6 / routing issues
            if "Cannot assign requested address" in error_msg or "Network is unreachable" in error_msg:
                detailed_msg = (
                    f"Connection failed: {error_msg}\n\n"
                    "💡 **Diagnosis:** This happens because your network does not support IPv6, which Supabase direct connections use.\n\n"
                    "✅ **Fix:** I've added an 'Advanced Settings' section below. Please use the **Connection Pooler Host** "
                    "(found in Supabase Dashboard > Settings > Database) with port **6543**."
                )
                return False, detailed_msg
                
            return False, f"Connection failed: {error_msg}"
        except Exception as e:
            self.is_connected = False
            return False, f"Connection failed: {str(e)}"
    
    def get_tables(self) -> tuple[List[str], str]:
        """Get list of all tables in the public schema"""
        if not self.is_connected or not self.engine:
            return [], "Not connected to database"
        
        try:
            # Get tables from public schema (default Supabase schema)
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            df = pd.read_sql(query, self.engine)
            tables = df['table_name'].tolist()
            return tables, f"Found {len(tables)} tables in public schema"
        except Exception as e:
            return [], f"Failed to fetch tables: {str(e)}"
    
    def get_table_info(self, table_name: str) -> tuple[pd.DataFrame, str]:
        """Get table schema/structure information"""
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            query = f"""
                SELECT 
                    column_name as "Column",
                    data_type as "Type",
                    is_nullable as "Nullable",
                    column_default as "Default"
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            df = pd.read_sql(query, self.engine)
            return df, f"Schema for table '{table_name}'"
        except Exception as e:
            return pd.DataFrame(), f"Failed to get table info: {str(e)}"
    
    def get_row_count(self, table_name: str) -> tuple[int, str]:
        """Get the number of rows in a table"""
        if not self.is_connected or not self.engine:
            return 0, "Not connected to database"
        
        try:
            query = f'SELECT COUNT(*) as count FROM public."{table_name}"'
            df = pd.read_sql(query, self.engine)
            count = int(df['count'].iloc[0])
            return count, f"Table '{table_name}' has {count:,} rows"
        except Exception as e:
            return 0, f"Failed to count rows: {str(e)}"
    
    def preview_table(self, table_name: str, limit: int = 100) -> tuple[pd.DataFrame, str]:
        """Preview table data with limited rows"""
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            query = f'SELECT * FROM public."{table_name}" LIMIT {limit}'
            df = pd.read_sql(query, self.engine)
            return df, f"Showing {len(df)} rows from '{table_name}'"
        except Exception as e:
            return pd.DataFrame(), f"Failed to preview table: {str(e)}"
    
    def import_table(self, table_name: str, limit: Optional[int] = None) -> tuple[pd.DataFrame, str]:
        """Import entire table as DataFrame"""
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            if limit:
                query = f'SELECT * FROM public."{table_name}" LIMIT {limit}'
            else:
                query = f'SELECT * FROM public."{table_name}"'
            
            df = pd.read_sql(query, self.engine)
            return df, f"Imported {len(df):,} rows and {len(df.columns)} columns from '{table_name}'"
        except Exception as e:
            return pd.DataFrame(), f"Failed to import table: {str(e)}"
    
    def import_query(self, query: str) -> tuple[pd.DataFrame, str]:
        """Import data using custom SQL query"""
        if not self.is_connected or not self.engine:
            return pd.DataFrame(), "Not connected to database"
        
        try:
            df = pd.read_sql(query, self.engine)
            return df, f"Query returned {len(df):,} rows and {len(df.columns)} columns"
        except Exception as e:
            return pd.DataFrame(), f"Failed to execute query: {str(e)}"
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information (without password)"""
        return self.connection_info.copy()
    
    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.is_connected = False
        self.connection_info = {}
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()


def render_database_connector_ui():
    """
    Render the database connection UI in Streamlit
    Returns the imported DataFrame if successful, None otherwise
    """
    from modules.utils import detect_column_types
    
    st.subheader("🔌 Connect to MySQL Database")
    
    # Connection form
    with st.form("mysql_connection_form"):
        st.markdown("Enter your MySQL database credentials:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input(
                "Host",
                value="localhost",
                help="Database server address (e.g., localhost, 192.168.1.100)"
            )
            port = st.number_input(
                "Port",
                value=3306,
                min_value=1,
                max_value=65535,
                help="MySQL default port is 3306"
            )
            database = st.text_input(
                "Database Name",
                help="Name of the database to connect to"
            )
        
        with col2:
            username = st.text_input(
                "Username",
                help="Database username"
            )
            password = st.text_input(
                "Password",
                type="password",
                help="Database password"
            )
            use_ssl = st.checkbox(
                "Use SSL Connection",
                value=False,
                help="Enable for secure connections (recommended for remote databases)"
            )
        
        connect_btn = st.form_submit_button("🔗 Connect to Database", type="primary", use_container_width=True)
    
    # Handle connection
    if connect_btn:
        if not all([host, database, username]):
            st.error("❌ Please fill in Host, Database Name, and Username")
            return None
        
        with st.spinner("Connecting to database..."):
            connector = MySQLConnector()
            success, message = connector.connect(host, port, database, username, password, use_ssl)
            
            if success:
                st.session_state.db_connector = connector
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                return None
    
    # If connected, show table selection
    if 'db_connector' in st.session_state and st.session_state.db_connector.is_connected:
        connector = st.session_state.db_connector
        
        st.divider()
        st.subheader("📋 Select Data to Import")
        
        # Get tables
        tables, msg = connector.get_tables()
        
        if not tables:
            st.warning(f"⚠️ {msg}")
            return None
        
        # Table selection
        selected_table = st.selectbox(
            "Select Table",
            options=tables,
            help="Choose a table to import"
        )
        
        if selected_table:
            # Show table info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                row_count, count_msg = connector.get_row_count(selected_table)
                st.info(f"📊 {count_msg}")
            
            with col2:
                if st.button("📖 View Schema"):
                    schema_df, schema_msg = connector.get_table_info(selected_table)
                    if not schema_df.empty:
                        st.dataframe(schema_df, use_container_width=True)
            
            # Preview table
            with st.expander("👁️ Preview Table Data", expanded=False):
                preview_rows = st.slider("Rows to preview", 10, 100, 50)
                preview_df, preview_msg = connector.preview_table(selected_table, preview_rows)
                if not preview_df.empty:
                    st.dataframe(preview_df, use_container_width=True)
                    st.caption(preview_msg)
            
            # Import options
            st.subheader("📥 Import Options")
            
            import_method = st.radio(
                "Import Method",
                options=["Import Entire Table", "Import with Row Limit", "Custom SQL Query"],
                horizontal=True
            )
            
            # Method-specific options
            row_limit = None
            custom_query = None
            
            if import_method == "Import with Row Limit":
                row_limit = st.number_input(
                    "Maximum Rows to Import",
                    min_value=100,
                    max_value=1000000,
                    value=min(10000, row_count) if row_count > 0 else 10000,
                    step=1000,
                    help="Limit the number of rows to import for large tables"
                )
            
            elif import_method == "Custom SQL Query":
                custom_query = st.text_area(
                    "SQL Query",
                    value=f"SELECT * FROM `{selected_table}` LIMIT 1000",
                    height=100,
                    help="Write a custom SQL query to import specific data"
                )
                st.caption("⚠️ Use caution with custom queries. Ensure proper syntax and avoid destructive operations.")
            
            # Import button
            if st.button("📥 Import Data", type="primary", use_container_width=True):
                with st.spinner("Importing data..."):
                    
                    if import_method == "Custom SQL Query" and custom_query:
                        df, msg = connector.import_query(custom_query)
                    else:
                        df, msg = connector.import_table(selected_table, row_limit)
                    
                    if not df.empty:
                        # Store in session state
                        st.session_state.dataset = df.copy()
                        st.session_state.original_dataset = df.copy()
                        
                        # Auto-detect column types
                        st.session_state.column_types = detect_column_types(df)
                        
                        # Clear previous analysis
                        st.session_state.column_analysis = {}
                        st.session_state.cleaning_history = {}
                        st.session_state.undo_stack = []
                        st.session_state.redo_stack = []
                        
                        st.success(f"✅ {msg}")
                        st.info("🔍 Column types automatically detected. Review them below and start your analysis!")
                        
                        # Show import summary
                        st.metric("Rows Imported", f"{len(df):,}")
                        st.metric("Columns Imported", len(df.columns))
                        
                        return df
                    else:
                        st.error(f"❌ {msg}")
                        return None
        
        # Disconnect button
        st.divider()
        if st.button("🔌 Disconnect from Database"):
            connector.disconnect()
            del st.session_state.db_connector
            st.success("Disconnected from database")
            st.rerun()
    
    return None


def render_supabase_connector_ui():
    """
    Render the Supabase connection UI in Streamlit
    Returns the imported DataFrame if successful, None otherwise
    """
    from modules.utils import detect_column_types
    
    st.subheader("🟢 Connect to Supabase")
    
    # Help section
    with st.expander("ℹ️ How to get your Supabase credentials", expanded=False):
        st.markdown("""
        **To connect to your Supabase database:**
        
        1. Go to your [Supabase Dashboard](https://supabase.com/dashboard)
        2. Select your project
        3. Go to **Settings** → **Database**
        4. Find your **Project URL** (e.g., `https://xxxx.supabase.co`)
        5. Find your **Database Password** (the one you set when creating the project)
        
        **Note:** Use the database password, NOT the API keys (anon/service_role).
        """)
    
    # Connection form
    with st.form("supabase_connection_form"):
        st.markdown("Enter your Supabase project credentials:")
        
        project_url = st.text_input(
            "Project URL / Reference",
            placeholder="https://xxxx.supabase.co or xxxx",
            help="Your Supabase project URL or 20-character project reference"
        )
        
        db_password = st.text_input(
            "Database Password",
            type="password",
            help="The database password you set when creating the project"
        )
        
        with st.expander("🛠️ Advanced / Connection Pooler Settings"):
            st.info("💡 If you see 'Cannot assign requested address' or connection failed error, try using the Connection Pooler details from your Supabase dashboard.")
            use_advanced = st.checkbox("Use Manual Connection Settings (Advanced)", value=False)
            
            adv_host = st.text_input(
                "Database Host", 
                placeholder="aws-0-us-east-1.pooler.supabase.com",
                help="Found in Supabase Dashboard > Settings > Database > Connection string > URI",
                disabled=not use_advanced
            )
            
            adv_port = st.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=6543,
                help="Default Direct: 5432, Connection Pooler: 6543",
                disabled=not use_advanced
            )
        
        connect_btn = st.form_submit_button("🔗 Connect to Supabase", type="primary", use_container_width=True)
    
    # Handle connection
    if connect_btn:
        if not all([project_url, db_password]):
            st.error("❌ Please provide both Project URL and Database Password")
            return None
        
        if use_advanced and not adv_host:
            st.error("❌ Please provide a Database Host for advanced connection")
            return None
        
        with st.spinner("Connecting to Supabase..."):
            connector = SupabaseConnector()
            
            if use_advanced:
                success, message = connector.connect(
                    project_url, 
                    db_password, 
                    custom_host=adv_host, 
                    custom_port=adv_port
                )
            else:
                success, message = connector.connect(project_url, db_password)
            
            if success:
                st.session_state.supabase_connector = connector
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                return None
    
    # If connected, show table selection
    if 'supabase_connector' in st.session_state and st.session_state.supabase_connector.is_connected:
        connector = st.session_state.supabase_connector
        
        st.divider()
        st.subheader("📋 Select Data to Import")
        
        # Get tables
        tables, msg = connector.get_tables()
        
        if not tables:
            st.warning(f"⚠️ {msg}")
            st.info("💡 Make sure you have tables in the 'public' schema of your Supabase database.")
            return None
        
        # Table selection
        selected_table = st.selectbox(
            "Select Table",
            options=tables,
            help="Choose a table to import",
            key="supabase_table_select"
        )
        
        if selected_table:
            # Show table info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                row_count, count_msg = connector.get_row_count(selected_table)
                st.info(f"📊 {count_msg}")
            
            with col2:
                if st.button("📖 View Schema", key="supabase_schema_btn"):
                    schema_df, schema_msg = connector.get_table_info(selected_table)
                    if not schema_df.empty:
                        st.dataframe(schema_df, use_container_width=True)
            
            # Preview table
            with st.expander("👁️ Preview Table Data", expanded=False):
                preview_rows = st.slider("Rows to preview", 10, 100, 50, key="supabase_preview_slider")
                preview_df, preview_msg = connector.preview_table(selected_table, preview_rows)
                if not preview_df.empty:
                    st.dataframe(preview_df, use_container_width=True)
                    st.caption(preview_msg)
            
            # Import options
            st.subheader("📥 Import Options")
            
            import_method = st.radio(
                "Import Method",
                options=["Import Entire Table", "Import with Row Limit", "Custom SQL Query"],
                horizontal=True,
                key="supabase_import_method"
            )
            
            # Method-specific options
            row_limit = None
            custom_query = None
            
            if import_method == "Import with Row Limit":
                row_limit = st.number_input(
                    "Maximum Rows to Import",
                    min_value=100,
                    max_value=1000000,
                    value=min(10000, row_count) if row_count > 0 else 10000,
                    step=1000,
                    help="Limit the number of rows to import for large tables",
                    key="supabase_row_limit"
                )
            
            elif import_method == "Custom SQL Query":
                custom_query = st.text_area(
                    "SQL Query",
                    value=f'SELECT * FROM public."{selected_table}" LIMIT 1000',
                    height=100,
                    help="Write a custom SQL query to import specific data",
                    key="supabase_custom_query"
                )
                st.caption("⚠️ Use PostgreSQL syntax. Tables are in the 'public' schema by default.")
            
            # Import button
            if st.button("📥 Import Data", type="primary", use_container_width=True, key="supabase_import_btn"):
                with st.spinner("Importing data from Supabase..."):
                    
                    if import_method == "Custom SQL Query" and custom_query:
                        df, msg = connector.import_query(custom_query)
                    else:
                        df, msg = connector.import_table(selected_table, row_limit)
                    
                    if not df.empty:
                        # Store in session state
                        st.session_state.dataset = df.copy()
                        st.session_state.original_dataset = df.copy()
                        
                        # Auto-detect column types
                        st.session_state.column_types = detect_column_types(df)
                        
                        # Clear previous analysis
                        st.session_state.column_analysis = {}
                        st.session_state.cleaning_history = {}
                        st.session_state.undo_stack = []
                        st.session_state.redo_stack = []
                        
                        st.success(f"✅ {msg}")
                        st.info("🔍 Column types automatically detected. Review them below and start your analysis!")
                        
                        # Show import summary
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows Imported", f"{len(df):,}")
                        with col2:
                            st.metric("Columns Imported", len(df.columns))
                        
                        return df
                    else:
                        st.error(f"❌ {msg}")
                        return None
        
        # Disconnect button
        st.divider()
        if st.button("🔌 Disconnect from Supabase", key="supabase_disconnect_btn"):
            connector.disconnect()
            del st.session_state.supabase_connector
            st.success("Disconnected from Supabase")
            st.rerun()
    
    return None
