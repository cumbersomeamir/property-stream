'use client';

import { useEffect, useState } from 'react';
import Papa from 'papaparse';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

interface GDDPData {
  district: string;
  gddp: number;
  gddpConstant: number;
  nddp: number;
  perCapitaIncome: number;
}

interface ClimateData {
  state: string;
  climateVulIndex: number;
  populationDensity: number;
  literacyRate: number;
  povertyRate: number;
  gdp: number;
  irrigatedArea: number;
  forestArea: number;
}

interface CoalData {
  year: number;
  sector: string;
  quantity: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];

export default function AllDataPage() {
  const [gddpData, setGddpData] = useState<GDDPData[]>([]);
  const [climateStateData, setClimateStateData] = useState<ClimateData[]>([]);
  const [coalYearly, setCoalYearly] = useState<any[]>([]);
  const [coalSectors, setCoalSectors] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      // Load Karnataka GDDP data
      const gddpResponse = await fetch('/data/state-sources/gdp/karnataka-gddp-2022-23.csv');
      const gddpText = await gddpResponse.text();
      Papa.parse(gddpText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const parsed = (results.data as any[])
            .filter(row => row.District && row.District !== 'State')
            .map(row => ({
              district: row.District,
              gddp: parseFloat(row['GDDP in Rs Cr (Current prices)']?.replace(/,/g, '') || '0'),
              gddpConstant: parseFloat(row['GDDP in Rs Cr (Constant prices) ']?.replace(/,/g, '') || '0'),
              nddp: parseFloat(row['Net District Domestic Product in Rs. Cr (Current Prices)']?.replace(/,/g, '') || '0'),
              perCapitaIncome: parseFloat(row['Per Capita Income in Rs. (Per Capita NDDP at Current Prices)']?.replace(/,/g, '') || '0'),
            }))
            .filter(d => d.gddp > 0);
          setGddpData(parsed);
        },
      });

      // Load Climate Vulnerability - State level
      const climateResponse = await fetch('/data/indiadataportal/gdp/Climate_Vulnerability_Indicators_1.csv');
      const climateText = await climateResponse.text();
      Papa.parse(climateText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const parsed = (results.data as any[])
            .filter(row => row.state_name && row.climate_vul_in && row.climate_vul_in !== '')
            .map(row => ({
              state: row.state_name,
              climateVulIndex: parseFloat(row.climate_vul_in || '0'),
              populationDensity: parseFloat(row.population_density || '0'),
              literacyRate: parseFloat(row.literacy_rate || '0'),
              povertyRate: parseFloat(row.poverty_rate || '0'),
              gdp: parseFloat(row.gdp_const_price || '0'),
              irrigatedArea: parseFloat(row.irrigated_area || '0'),
              forestArea: parseFloat(row.forest_land_area || '0'),
            }))
            .filter(d => d.climateVulIndex > 0);
          setClimateStateData(parsed);
        },
      });

      // Load Coal Usage - Time series
      const coalResponse = await fetch('/data/indiadataportal/gdp/Coal_Usage_in_India_1.csv');
      const coalText = await coalResponse.text();
      Papa.parse(coalText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const data = results.data as any[];
          
          // Yearly aggregation
          const yearly = data
            .filter(row => row.year && row.quantity)
            .reduce((acc: any, row: any) => {
              const year = parseInt(row.year);
              const qty = parseFloat(row.quantity || '0');
              const existing = acc.find((item: any) => item.year === year);
              if (existing) {
                existing.quantity += qty;
              } else {
                acc.push({ year, quantity: qty });
              }
              return acc;
            }, [])
            .sort((a: any, b: any) => a.year - b.year);
          setCoalYearly(yearly);

          // Sector aggregation
          const sectors = data
            .filter(row => row.sector && row.quantity)
            .reduce((acc: any, row: any) => {
              const sector = row.sector;
              const qty = parseFloat(row.quantity || '0');
              const existing = acc.find((item: any) => item.sector === sector);
              if (existing) {
                existing.quantity += qty;
              } else {
                acc.push({ sector, quantity: qty });
              }
              return acc;
            }, [])
            .sort((a: any, b: any) => b.quantity - a.quantity);
          setCoalSectors(sectors);
        },
      });

      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl text-red-600">Error: {error}</div>
      </div>
    );
  }

  // Calculate insights
  const topGDDP = [...gddpData].sort((a, b) => b.gddp - a.gddp).slice(0, 10);
  const topPerCapita = [...gddpData].sort((a, b) => b.perCapitaIncome - a.perCapitaIncome).slice(0, 12);
  const bottomPerCapita = [...gddpData].sort((a, b) => a.perCapitaIncome - b.perCapitaIncome).slice(0, 8);
  const gddpVsIncome = gddpData.filter(d => d.gddp > 0 && d.perCapitaIncome > 0).slice(0, 15);
  
  const highVulnerabilityStates = [...climateStateData]
    .filter(d => d.climateVulIndex > 0.55)
    .sort((a, b) => b.climateVulIndex - a.climateVulIndex)
    .slice(0, 10);
  
  const vulnerabilityVsPoverty = climateStateData
    .filter(d => d.climateVulIndex > 0 && d.povertyRate > 0)
    .slice(0, 15);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 text-black">Property Stream - Data Analysis</h1>
        <p className="text-gray-600 mb-8">Comprehensive visualization of available datasets</p>

        {/* ECONOMIC INDICATORS SECTION */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">📊 Economic Performance Indicators</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Top 10 Districts by GDDP (₹ Cr)</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={topGDDP} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="district"
                    angle={-60}
                    textAnchor="end"
                    height={100}
                    fontSize={10}
                    tick={{ fontSize: 9 }}
                  />
                  <YAxis tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(value: number) => `₹${(value / 1000).toFixed(1)}k Cr`} contentStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="gddp" fill="#0088FE" name="GDDP (₹ Cr)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">GDDP: Current vs Constant Prices</h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={topGDDP} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="district"
                    angle={-60}
                    textAnchor="end"
                    height={100}
                    fontSize={10}
                    tick={{ fontSize: 9 }}
                  />
                  <YAxis tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Area type="monotone" dataKey="gddp" stackId="1" stroke="#8884d8" fill="#8884d8" name="Current Prices" />
                  <Area type="monotone" dataKey="gddpConstant" stackId="1" stroke="#82ca9d" fill="#82ca9d" name="Constant Prices" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3">Per Capita Income Distribution (Top 12 Districts)</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={topPerCapita} margin={{ top: 20, right: 30, left: 20, bottom: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="district"
                  angle={-60}
                  textAnchor="end"
                  height={130}
                  fontSize={10}
                  tick={{ fontSize: 9 }}
                />
                <YAxis tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`} width={80} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value: number) => `₹${value.toLocaleString()}`} contentStyle={{ fontSize: '12px' }} />
                <Bar dataKey="perCapitaIncome" fill="#00C49F" name="Per Capita Income" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">GDDP vs Per Capita Income Correlation</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={gddpVsIncome} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="gddp"
                    name="GDDP"
                    unit=" Cr"
                    tickFormatter={(value) => `₹${(value / 100000).toFixed(0)}L`}
                    label={{ value: 'GDDP (₹ Cr)', position: 'insideBottom', offset: -5, style: { fontSize: '12px' } }}
                  />
                  <YAxis
                    type="number"
                    dataKey="perCapitaIncome"
                    name="Per Capita Income"
                    tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`}
                    label={{ value: 'Per Capita Income (₹)', angle: -90, position: 'insideLeft', style: { fontSize: '12px' } }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: '12px' }} />
                  <Scatter name="Districts" data={gddpVsIncome} fill="#8884d8">
                    {gddpVsIncome.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Bottom 8 Districts by Per Capita Income</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={bottomPerCapita} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="district"
                    angle={-60}
                    textAnchor="end"
                    height={100}
                    fontSize={10}
                    tick={{ fontSize: 9 }}
                  />
                  <YAxis tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`} width={80} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(value: number) => `₹${value.toLocaleString()}`} contentStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="perCapitaIncome" fill="#FF8042" name="Per Capita Income" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* CLIMATE & RISK INDICATORS */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">🌍 Climate Vulnerability & Risk Assessment</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Top 10 States by Climate Vulnerability Index</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={highVulnerabilityStates} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="state"
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    fontSize={10}
                    tick={{ fontSize: 9 }}
                  />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="climateVulIndex" fill="#FF8042" name="Vulnerability Index" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Climate Vulnerability vs Poverty Rate</h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart data={vulnerabilityVsPoverty} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="climateVulIndex"
                    name="Climate Vulnerability"
                    domain={[0, 1]}
                    label={{ value: 'Climate Vulnerability Index', position: 'insideBottom', offset: -5, style: { fontSize: '12px' } }}
                  />
                  <YAxis
                    type="number"
                    dataKey="povertyRate"
                    name="Poverty Rate"
                    label={{ value: 'Poverty Rate (%)', angle: -90, position: 'insideLeft', style: { fontSize: '12px' } }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: '12px' }} />
                  <Scatter name="States" data={vulnerabilityVsPoverty} fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Climate Vulnerability vs Literacy Rate</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={climateStateData.slice(0, 15)} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="state"
                    angle={-60}
                    textAnchor="end"
                    height={100}
                    fontSize={9}
                    tick={{ fontSize: 8 }}
                  />
                  <YAxis yAxisId="left" domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="climateVulIndex"
                    stroke="#FF8042"
                    name="Climate Index"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="literacyRate"
                    stroke="#00C49F"
                    name="Literacy (%)"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Infrastructure: Forest Area Distribution</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={climateStateData.slice(0, 12)} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="state"
                    angle={-60}
                    textAnchor="end"
                    height={100}
                    fontSize={9}
                    tick={{ fontSize: 8 }}
                  />
                  <YAxis tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(value: number) => `${(value / 1000).toFixed(1)}k sq km`} contentStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="forestArea" fill="#00C49F" name="Forest Area (sq km)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* ENERGY & ECONOMIC ACTIVITY */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">⚡ Energy Consumption & Economic Activity</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Coal Consumption Trend Over Time</h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={coalYearly} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="year"
                    label={{ value: 'Year', position: 'insideBottom', offset: -5, style: { fontSize: '12px' } }}
                  />
                  <YAxis tickFormatter={(value) => `${value.toFixed(0)}`} label={{ value: 'Quantity (MT)', angle: -90, position: 'insideLeft', style: { fontSize: '12px' } }} />
                  <Tooltip formatter={(value: number) => `${value.toFixed(1)} MT`} contentStyle={{ fontSize: '12px' }} />
                  <Area type="monotone" dataKey="quantity" stroke="#8884d8" fill="#8884d8" name="Coal (MT)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Coal Usage by Sector Distribution</h3>
              <ResponsiveContainer width="100%" height={350}>
                <PieChart>
                  <Pie
                    data={coalSectors}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="quantity"
                  >
                    {coalSectors.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number, name, props) => {
                      const sector = coalSectors[props.payload?.index]?.sector || '';
                      return [`${sector}: ${value.toFixed(1)} MT`, 'Quantity'];
                    }}
                    contentStyle={{ fontSize: '12px' }}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '20px' }}
                    formatter={(value, entry) => coalSectors[entry.payload?.index]?.sector || value}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-3">Coal Usage by Sector (Bar Chart)</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={coalSectors} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="sector"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  fontSize={11}
                  tick={{ fontSize: 10 }}
                />
                <YAxis tickFormatter={(value) => `${value.toFixed(0)}`} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value: number) => `${value.toFixed(1)} MT`} contentStyle={{ fontSize: '12px' }} />
                <Bar dataKey="quantity" fill="#0088FE" name="Quantity (MT)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* KEY INSIGHTS SUMMARY */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">💡 Key Insights for Property Investment</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="border-l-4 border-blue-500 pl-4 bg-blue-50 p-4 rounded">
              <h3 className="font-semibold text-gray-700 mb-2">Highest Per Capita Income</h3>
              <p className="text-2xl font-bold text-blue-600">
                {topPerCapita[0]?.district || 'N/A'}
              </p>
              <p className="text-sm text-gray-600">
                ₹{topPerCapita[0]?.perCapitaIncome.toLocaleString() || '0'}
              </p>
              <p className="text-xs text-gray-500 mt-1">Strong purchasing power = high property demand</p>
            </div>

            <div className="border-l-4 border-green-500 pl-4 bg-green-50 p-4 rounded">
              <h3 className="font-semibold text-gray-700 mb-2">Highest GDDP District</h3>
              <p className="text-2xl font-bold text-green-600">
                {topGDDP[0]?.district || 'N/A'}
              </p>
              <p className="text-sm text-gray-600">
                ₹{(topGDDP[0]?.gddp / 1000).toFixed(1)}k Cr
              </p>
              <p className="text-xs text-gray-500 mt-1">Economic powerhouse = stable growth</p>
            </div>

            <div className="border-l-4 border-red-500 pl-4 bg-red-50 p-4 rounded">
              <h3 className="font-semibold text-gray-700 mb-2">Highest Climate Risk</h3>
              <p className="text-2xl font-bold text-red-600">
                {highVulnerabilityStates[0]?.state || 'N/A'}
              </p>
              <p className="text-sm text-gray-600">
                Index: {highVulnerabilityStates[0]?.climateVulIndex.toFixed(3) || '0'}
              </p>
              <p className="text-xs text-gray-500 mt-1">Higher risk = consider insurance & location</p>
            </div>

            <div className="border-l-4 border-purple-500 pl-4 bg-purple-50 p-4 rounded">
              <h3 className="font-semibold text-gray-700 mb-2">Coal Consumption Trend</h3>
              <p className="text-2xl font-bold text-purple-600">
                {coalYearly.length > 0 ? (coalYearly[coalYearly.length - 1].quantity / coalYearly[0].quantity - 1) * 100 > 0 ? '↑' : '↓' : 'N/A'}
              </p>
              <p className="text-sm text-gray-600">
                {coalYearly.length > 0 ? `${((coalYearly[coalYearly.length - 1].quantity / coalYearly[0].quantity - 1) * 100).toFixed(1)}%` : '0%'} change
              </p>
              <p className="text-xs text-gray-500 mt-1">Economic activity indicator</p>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-3">Investment Recommendations Based on Data</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li>• <strong>High Value Targets:</strong> Districts with per capita income &gt; ₹300k show strong property demand potential</li>
              <li>• <strong>Growth Opportunities:</strong> Districts with high GDDP but lower per capita income may offer better ROI</li>
              <li>• <strong>Risk Mitigation:</strong> Consider climate vulnerability index - lower scores indicate safer long-term investments</li>
              <li>• <strong>Economic Stability:</strong> Higher literacy rates and lower poverty rates correlate with stable property markets</li>
              <li>• <strong>Infrastructure:</strong> Areas with better forest coverage and irrigation show sustainable development patterns</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
