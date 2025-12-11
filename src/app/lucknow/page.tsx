'use client';

import { useEffect, useState } from 'react';
import { parseCSV } from '@/lib/csvParser';
import ChartWrapper from '@/components/ChartWrapper';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area
} from 'recharts';

const COLORS = ['#2F3D36', '#C2B8A3', '#E6DED2', '#A46B47', '#8B7355', '#6B5B4F'];

export default function LucknowPage() {
  const [data, setData] = useState<any>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      const datasets: any = {};
      
      try {
        // Load all datasets
        datasets.demographics = await parseCSV('/data/D01_DemographicProfile_Lucknow (1).csv');
        datasets.unemployment = await parseCSV('/data/D02_UnemploymentRate_Lucknow_2.csv');
        datasets.households = await parseCSV('/data/D03_Households_Lucknow_0.csv');
        datasets.environment = await parseCSV('/data/D04_Environment_lucknow_1.csv');
        datasets.publicAmenities = await parseCSV('/data/D05_PublicAmenities_lucknow_1.csv');
        datasets.communityFacilities = await parseCSV('/data/D06_CommunityFacilities_Lucknow_1.csv');
        datasets.health = await parseCSV('/data/D08_HealthInfrastructure_lucknow_1.csv');
        datasets.mortality = await parseCSV('/data/D09_Mortality_Lucknow_7.csv');
        datasets.propertyTax = await parseCSV('/data/D21_PropertyTax_lucknow_1.csv');
        datasets.housing = await parseCSV('/data/D25_Housing_SlumPopulation_Lucknow.csv');
        datasets.buses = await parseCSV('/data/D36_Buses_Lucknow.csv');
        datasets.vehicles = await parseCSV('/data/D40_VehicleRegistration_lucknow_1.csv');
        datasets.governance = await parseCSV('/data/D44_Governance_Lucknow_1.csv');
        datasets.roads = await parseCSV('/data/D23_Conditionofroads_lucknow.csv');
        datasets.streetLights = await parseCSV('/data/D24_StreetLights_Lucknow_2.csv');
        datasets.waterTax = await parseCSV('/data/D10_WaterTax_Lucknow_1_1.csv');
        datasets.publicToilet = await parseCSV('/data/D11_PublicToilet_Lucknow_2.csv');
        datasets.solidWaste = await parseCSV('/data/D17_SolidWasteCollectionVehicle_Lucknow_1.csv');
        datasets.wasteProcessing = await parseCSV('/data/D18_SolidWasteProcessing_Lucknow.csv');
        datasets.openSpaces = await parseCSV('/data/D29_openspaces_lucknow_2.csv');
        datasets.busTrips = await parseCSV('/data/D37_EarningsBusTrips_Lucknow.csv');
        datasets.transportAccess = await parseCSV('/data/D38_PublicTransportAccess_Lucknow_2.csv');
        datasets.injuries = await parseCSV('/data/D41_Injuries_Fatilities_Lucknow_1.csv');
        datasets.digital = await parseCSV('/data/D45_DigitalAvailability_Lucknow_1.csv');
        datasets.diseases = await parseCSV('/data/D46_Diseases_Lucknow_1.csv');
        datasets.vatGst = await parseCSV('/data/D48_VAT_GST_Lucknow_1.csv');
        datasets.digitalPayments = await parseCSV('/data/D50_DigitalPayments_Lucknow_1.csv');
        datasets.intersections = await parseCSV('/data/D35_Signalized_Intersections_lucknow_1.csv');
        datasets.heritage = await parseCSV('/data/D31_Cultural_Heritage_lucknow.csv');
        datasets.areaBifurcation = await parseCSV('/data/D33_AreaBifurcationdata_lucknow.csv');
        
        setData(datasets);
      } catch (error) {
        console.error('Error loading data:', error);
      } finally {
        setLoading(false);
      }
    }
    
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading data...</div>
      </div>
    );
  }

  // Process demographics data
  const demographics = data.demographics?.[0];
  const populationData = demographics ? [
    { name: 'Male', value: parseFloat(demographics['Population - Male (in thousands)']) || 0 },
    { name: 'Female', value: parseFloat(demographics['Population - female (in thousands)']) || 0 }
  ] : [];

  // Process unemployment data
  const unemployment = data.unemployment?.[0];
  const employmentData = unemployment ? [
    { name: 'Employed', value: parseInt(unemployment['No. of employed persons']) || 0 },
    { name: 'Unemployed', value: parseInt(unemployment['No. of unemployed persons (seeking or available for work)']) || 0 }
  ] : [];

  // Process households by zone
  const householdsByZone: any = {};
  data.households?.forEach((row: any) => {
    const zone = row['Zone Name']?.trim() || 'Unknown';
    const households = parseInt(row['Total no of Households']) || 0;
    if (!householdsByZone[zone]) {
      householdsByZone[zone] = 0;
    }
    householdsByZone[zone] += households;
  });
  const householdChartData = Object.entries(householdsByZone).map(([zone, count]) => ({
    zone: zone.replace('Zone', 'Z').replace(/\s+/g, ''),
    households: count
  })).sort((a, b) => b.households - a.households).slice(0, 8);

  // Process environment data - PM2.5 over time
  const environmentData = data.environment
    ?.filter((row: any) => row['Monthly mean/average concentration - PM2.5'] && row['Monthly mean/average concentration - PM2.5'] !== 'NA')
    .map((row: any) => ({
      month: row['Month -Year'],
      pm25: parseFloat(row['Monthly mean/average concentration - PM2.5']) || 0,
      no2: parseFloat(row['Monthly mean concentration - NO2']) || 0,
      so2: parseFloat(row['Monthly mean concentration - SO2']) || 0
    }))
    .reverse() || [];

  // Process property tax data
  const propertyTaxData = data.propertyTax?.map((row: any) => {
    const zone = `Zone ${row['Zone Name']}`;
    const residential2017 = parseFloat(row['2017-18 - Property Tax Collection (in crores) - Residential']) || 0;
    const commercial2017 = parseFloat(row['2017-18 - Property Tax Collection (in crores) - Commercial']) || 0;
    return {
      zone,
      residential: residential2017,
      commercial: commercial2017,
      total: residential2017 + commercial2017
    };
  }).filter((d: any) => d.total > 0) || [];

  // Process health infrastructure
  const healthData = data.health?.filter((row: any) => 
    row['Number of Beds in facility type'] && row['Number of Beds in facility type'] !== 'NA'
  ).map((row: any) => ({
    facility: row['Facility Name']?.substring(0, 20) || 'Unknown',
    beds: parseInt(row['Number of Beds in facility type']) || 0,
    zone: row['Zone Name']
  })).sort((a: any, b: any) => b.beds - a.beds).slice(0, 15) || [];

  // Process street lights by zone
  const streetLightsByZone: any = {};
  data.streetLights?.forEach((row: any) => {
    const zone = row['Zone Name'] || 'Unknown';
    const poles = parseInt(row['Number of Poles']) || 0;
    if (!streetLightsByZone[zone]) {
      streetLightsByZone[zone] = 0;
    }
    streetLightsByZone[zone] += poles;
  });
  const streetLightsData = Object.entries(streetLightsByZone).map(([zone, count]) => ({
    zone: zone.replace('Zone', 'Z'),
    poles: count
  }));

  // Process vehicle registration
  const vehicleData = data.vehicles?.[0];
  const vehicleChartData = vehicleData ? [
    { name: 'Buses', value: parseInt(vehicleData['No. of Registrations - Buses - 2017-18']) || 0 },
    { name: 'LMV', value: parseInt(vehicleData['No. of Registrations - Light Motor Vehicles - 2017-18']) || 0 },
    { name: 'Goods Carrier', value: parseInt(vehicleData['No. of Registrations - Goods Carrier Vehicles - 2017-18']) || 0 },
    { name: 'Two-Wheeler', value: parseInt(vehicleData['No. of Registrations - Two-Wheeler Vehicles - 2017-18']) || 0 }
  ].filter(v => v.value > 0) : [];

  // Process governance - voter turnout
  const governanceData = data.governance?.map((row: any) => {
    const registered = parseInt(row['Total no. of registered voters']) || 0;
    const polled = parseInt(row['No. of votes polled in the last municipal election']) || 0;
    const turnout = registered > 0 ? (polled / registered) * 100 : 0;
    return {
      ward: row['Ward Name']?.substring(0, 15) || 'Unknown',
      registered,
      polled,
      turnout: Math.round(turnout)
    };
  }).sort((a: any, b: any) => b.turnout - a.turnout).slice(0, 20) || [];

  // Process water tax
  const waterTaxData = data.waterTax?.filter((row: any) => 
    row['Zone Name'] && !row['Zone Name'].includes('Govt')
  ).map((row: any) => {
    const zone = `Zone ${row['Zone Name']}`;
    const collected2017 = parseFloat(row['Water tax collected (in INR lakhs)-Total-2017-18']) || 0;
    const billed2017 = parseFloat(row['Water tax billed (in INR lakhs)-Total-2017-18']) || 0;
    const efficiency = billed2017 > 0 ? (collected2017 / billed2017) * 100 : 0;
    return { zone, collected: collected2017, billed: billed2017, efficiency: Math.round(efficiency) };
  }) || [];

  // Process mortality data
  const mortalityData = data.mortality?.[0];
  const mortalityChartData = mortalityData ? [
    { year: '2013-14', births: parseFloat(mortalityData['Total No. of Live Births (in Thousands)201314']) || 0, deaths: parseFloat(mortalityData['No. of Deaths  Total (in Thousands)201314']) || 0 },
    { year: '2014-15', births: parseFloat(mortalityData['Total No. of Live Births (in Thousands)201415']) || 0, deaths: parseFloat(mortalityData['No. of Deaths  Total (in Thousands)201415']) || 0 },
    { year: '2015-16', births: parseFloat(mortalityData['Total No. of Live Births (in Thousands)201516']) || 0, deaths: parseFloat(mortalityData['No. of Deaths  Total (in Thousands)201516']) || 0 },
    { year: '2016-17', births: parseFloat(mortalityData['Total No. of Live Births (in Thousands)201617']) || 0, deaths: parseFloat(mortalityData['No. of Deaths  Total (in Thousands)201617']) || 0 },
    { year: '2017-18', births: parseFloat(mortalityData['Total No. of Live Births (in Thousands)201718']) || 0, deaths: parseFloat(mortalityData['No. of Deaths  Total (in Thousands)201718']) || 0 }
  ].filter(d => d.births > 0 || d.deaths > 0) : [];

  // Process public toilet data
  const toiletData = data.publicToilet?.map((row: any) => ({
    zone: row['Zone Name'] || 'Unknown',
    households: parseInt(row['Total number of households (HH)']) || 0,
    sewerage: parseInt(row['HH part of the city sewerage network']) || 0,
    coverage: parseInt(row['Total number of households (HH)']) > 0 
      ? Math.round((parseInt(row['HH part of the city sewerage network']) / parseInt(row['Total number of households (HH)'])) * 100)
      : 0
  })) || [];

  // Process diseases data
  const diseasesData = data.diseases?.map((row: any) => ({
    year: row['Year'],
    vector: parseInt(row['Number of cases of persons affected by vector borne diseases']) || 0,
    water: parseInt(row['Number of cases of persons affected by water borne diseases']) || 0
  })) || [];

  // Process VAT/GST data
  const vatGstData = data.vatGst?.map((row: any) => ({
    year: row['Year'],
    collection: parseFloat(row['Total VAT/GST collection during the year (in INR)']) || 0
  })) || [];

  // Process bus types
  const busData = data.buses?.map((row: any) => ({
    type: row['Type of bus (AC / Non AC)'] || 'Unknown',
    count: parseInt(row['No. of buses of that type']) || 0
  })).filter((b: any) => b.count > 0) || [];

  // Process public amenities summary
  const amenities = data.publicAmenities?.[0];

  // Process property tax trends (multi-year)
  const propertyTaxTrendsRaw = data.propertyTax?.map((row: any) => {
    const years = ['2013-14', '2014-15', '2015-16', '2016-17', '2017-18'];
    const zone = `Zone ${row['Zone Name']}`;
    const trendData: any = { zone };
    years.forEach(year => {
      const key = `${year} - Property Tax Collection (in crores) - Residential`;
      trendData[year] = parseFloat(row[key]) || 0;
    });
    return trendData;
  }).filter((d: any) => Object.keys(d).some(k => k !== 'zone' && d[k] > 0)) || [];

  // Process water tax trends
  const waterTaxTrends = data.waterTax?.filter((row: any) => 
    row['Zone Name'] && !row['Zone Name'].includes('Govt')
  ).map((row: any) => {
    const zone = `Zone ${row['Zone Name']}`;
    return {
      zone,
      '2013-14': parseFloat(row['Water tax collected (in INR lakhs)-Total-2013-14']) || 0,
      '2014-15': parseFloat(row['Water tax collected (in INR lakhs)-Total-2014-15']) || 0,
      '2015-16': parseFloat(row['Water tax collected (in INR lakhs)-Total-2015-16']) || 0,
      '2016-17': parseFloat(row['Water tax collected (in INR lakhs)-Total-2016-17']) || 0,
      '2017-18': parseFloat(row['Water tax collected (in INR lakhs)-Total-2017-18']) || 0
    };
  }) || [];

  // Process digital payments
  const digitalPaymentsData = data.digitalPayments?.filter((row: any) => 
    row['Month'] && row['Amount collected from citizens by Municipal Corporation / smart city for various services such as Property Tax etc. - Total Collection']
  ).map((row: any) => {
    const total = parseFloat(row['Amount collected from citizens by Municipal Corporation / smart city for various services such as Property Tax etc. - Total Collection']) || 0;
    const digital = parseFloat(row['Amount collected from citizens by Municipal Corporation / smart city for various services such as Property Tax etc. - Amount Collected as digital payment (non-cash, non cheque)']) || 0;
    return {
      month: row['Month'],
      total: total / 1000000, // Convert to crores
      digital: digital / 1000000,
      percentage: total > 0 ? (digital / total) * 100 : 0
    };
  }).reverse() || [];

  // Process injuries/fatalities trends
  const injuriesData = data.injuries?.[0];
  const roadSafetyData = injuriesData ? [
    {
      year: '2016',
      injuries: parseInt(injuriesData['2016 - Total Injuries']) || 0,
      fatalities: parseInt(injuriesData['2016 - Total Fatalities']) || 0
    },
    {
      year: '2017',
      injuries: parseInt(injuriesData['2017 - Total Injuries']) || 0,
      fatalities: parseInt(injuriesData['2017 - Total Fatalities']) || 0
    },
    {
      year: '2018',
      injuries: parseInt(injuriesData['2018 - Total Injuries']) || 0,
      fatalities: parseInt(injuriesData['2018 - Total Fatalities']) || 0
    }
  ].filter(d => d.injuries > 0 || d.fatalities > 0) : [];

  // Process health infrastructure by zone
  const healthByZone: any = {};
  data.health?.forEach((row: any) => {
    const zone = row['Zone Name'] || 'Unknown';
    if (!healthByZone[zone]) {
      healthByZone[zone] = { facilities: 0, totalBeds: 0 };
    }
    healthByZone[zone].facilities += 1;
    const beds = parseInt(row['Number of Beds in facility type']) || 0;
    healthByZone[zone].totalBeds += beds;
  });
  const healthZoneData = Object.entries(healthByZone).map(([zone, stats]: [string, any]) => ({
    zone: zone.replace('Zone', 'Z').replace(/\s+/g, ''),
    facilities: stats.facilities,
    beds: stats.totalBeds
  }));

  // Process top wards by households
  const topWards = data.households?.map((row: any) => ({
    ward: row['Ward Name']?.substring(0, 25) || 'Unknown',
    zone: row['Zone Name']?.replace('Zone', 'Z') || 'Unknown',
    households: parseInt(row['Total no of Households']) || 0
  })).sort((a: any, b: any) => b.households - a.households).slice(0, 20) || [];

  // Process solid waste collection coverage
  const wasteCoverage = data.solidWaste?.map((row: any) => {
    const households = parseInt(row['Total No. of households / establishments']) || 0;
    const covered = parseInt(row['Total no. of households and establishments covered through doorstep collection']) || 0;
    return {
      ward: row['Ward Name']?.substring(0, 20) || 'Unknown',
      households,
      covered,
      coverage: households > 0 ? Math.round((covered / households) * 100) : 0
    };
  }).sort((a: any, b: any) => b.coverage - a.coverage).slice(0, 15) || [];

  // Process heritage sites
  const heritageData = data.heritage?.map((row: any) => ({
    name: row['Name of heritage']?.substring(0, 25) || 'Unknown',
    zone: row['Zone Name'] || 'Unknown',
    nature: row['Nature of heritage (open space, monuments, street etc.)'] || 'Unknown',
    use: row['Heritage use'] || 'Unknown'
  })) || [];

  // Process intersections
  const intersectionData = data.intersections?.reduce((acc: any, row: any) => {
    const zone = row['Zone Name'] || 'Unknown';
    if (!acc[zone]) {
      acc[zone] = { total: 0, operational: 0 };
    }
    acc[zone].total += 1;
    if (row['Total number of operational signalized intersections'] === 'Yes') {
      acc[zone].operational += 1;
    }
    return acc;
  }, {});
  const intersectionChartData = Object.entries(intersectionData || {}).map(([zone, stats]: [string, any]) => ({
    zone: zone.replace('Zone', 'Z').replace(/\s+/g, ''),
    total: stats.total,
    operational: stats.operational,
    operationalRate: stats.total > 0 ? Math.round((stats.operational / stats.total) * 100) : 0
  }));

  // Process open spaces by zone
  const openSpacesByZone: any = {};
  data.openSpaces?.forEach((row: any) => {
    const zone = row['Zone Name'] || 'Unknown';
    const area = parseFloat(row['Open space area (in sqkm)']) || 0;
    if (!openSpacesByZone[zone]) {
      openSpacesByZone[zone] = { count: 0, totalArea: 0 };
    }
    openSpacesByZone[zone].count += 1;
    openSpacesByZone[zone].totalArea += area;
  });
  const openSpacesData = Object.entries(openSpacesByZone).map(([zone, stats]: [string, any]) => ({
    zone: zone.replace('Zone', 'Z').replace(/\s+/g, ''),
    count: stats.count,
    area: Math.round(stats.totalArea)
  }));

  // Process bus revenue efficiency
  const busEfficiencyData = data.busTrips?.filter((row: any) => 
    row['Year'] && row['Public Transport - Revenue per km']
  ).map((row: any) => {
    const costPerKm = parseFloat(row['Public Transport - Average cost per km (daily)']) || 0;
    const revenuePerKm = parseFloat(row['Public Transport - Revenue per km']) || 0;
    return {
      year: row['Year'],
      costPerKm,
      revenuePerKm,
      profitPerKm: revenuePerKm - costPerKm,
      efficiency: costPerKm > 0 ? ((revenuePerKm / costPerKm) * 100) : 0
    };
  }) || [];

  // Process property tax collection efficiency
  const propertyTaxEfficiency = data.propertyTax?.map((row: any) => {
    const zone = `Zone ${row['Zone Name']}`;
    const collected = parseFloat(row['2017-18 - Property Tax Collection (in crores) - Residential']) || 0;
    const demand = parseFloat(row['2017-18 - Property Tax Demand (in crores) - Residential']) || 0;
    const commercialCollected = parseFloat(row['2017-18 - Property Tax Collection (in crores) - Commercial']) || 0;
    const commercialDemand = parseFloat(row['2017-18 - Property Tax Demand (in crores) - Commercial']) || 0;
    const totalCollected = collected + commercialCollected;
    const totalDemand = demand + commercialDemand;
    return {
      zone,
      collected: totalCollected,
      demand: totalDemand,
      efficiency: totalDemand > 0 ? Math.round((totalCollected / totalDemand) * 100) : 0
    };
  }).filter((d: any) => d.demand > 0) || [];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Lucknow - Comprehensive Data Analysis</h1>
              <p className="text-lg text-gray-600">Property Investment Insights & City Statistics</p>
            </div>
            <div className="flex gap-4">
              <a href="/predict" className="text-green-600 hover:text-green-800 underline font-semibold">📊 Investment Predictor →</a>
              <a href="/" className="text-blue-600 hover:text-blue-800 underline">← Back to Home</a>
            </div>
          </div>
        </div>

        {/* Demographics Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Demographics & Population
          </h2>
          
          <ChartWrapper
            title="Population Distribution by Gender"
            source="D01_DemographicProfile_Lucknow"
            insight={`Total Population: ${(parseFloat(demographics?.['Total Population (in thousands)']) || 0).toLocaleString()}K | Area: ${demographics?.['Area (in sq km)'] || 'N/A'} sq km`}
          >
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={populationData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {populationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Employment Status"
            source="D02_UnemploymentRate_Lucknow"
            insight={`Unemployment Rate: ${unemployment ? ((parseInt(unemployment['No. of unemployed persons (seeking or available for work)']) / parseInt(unemployment['Total labour force in the city (age 15-59) [Employed + Unemployed Persons)'])) * 100).toFixed(2) : 'N/A'}%`}
          >
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={employmentData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {employmentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => value.toLocaleString()} />
              </PieChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Households Distribution by Zone"
            source="D03_Households_Lucknow"
            insight="Zone 4 (Chinhhat) has the highest number of households, indicating dense residential development"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={householdChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip formatter={(value: number) => value.toLocaleString()} />
                <Legend />
                <Bar dataKey="households" fill="#2F3D36" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Environment Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Environment & Air Quality
          </h2>
          
          <ChartWrapper
            title="Air Quality Trends (PM2.5, NO2, SO2)"
            source="D04_Environment_lucknow"
            insight="PM2.5 shows seasonal variation with peaks in winter months (Dec-Jan). Monsoon months show cleaner air."
          >
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={environmentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="pm25" stroke="#e74c3c" name="PM2.5 (µg/m³)" strokeWidth={2} />
                <Line type="monotone" dataKey="no2" stroke="#3498db" name="NO2 (µg/m³)" strokeWidth={2} />
                <Line type="monotone" dataKey="so2" stroke="#2ecc71" name="SO2 (µg/m³)" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Infrastructure Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Infrastructure & Amenities
          </h2>
          
          <ChartWrapper
            title="Top 15 Healthcare Facilities by Bed Capacity"
            source="D08_HealthInfrastructure_lucknow"
            insight="Healthcare infrastructure is well distributed, with some facilities having 100+ bed capacity"
          >
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={healthData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="facility" type="category" width={150} />
                <Tooltip formatter={(value: number) => `${value} beds`} />
                <Legend />
                <Bar dataKey="beds" fill="#A46B47" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Street Lighting Infrastructure by Zone"
            source="D24_StreetLights_Lucknow"
            insight="Adequate street lighting indicates better safety and livability. Zone 2 shows highest coverage."
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={streetLightsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis />
                <Tooltip formatter={(value: number) => `${value.toLocaleString()} poles`} />
                <Legend />
                <Bar dataKey="poles" fill="#C2B8A3" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Vehicle Registration by Type (2017-18)"
            source="D40_VehicleRegistration_lucknow"
            insight="Two-wheeler dominance indicates need for better public transport infrastructure"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={vehicleChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value: number) => value.toLocaleString()} />
                <Legend />
                <Bar dataKey="value" fill="#2F3D36" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Property & Revenue Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Property Tax & Revenue
          </h2>
          
          <ChartWrapper
            title="Property Tax Collection by Zone (2017-18)"
            source="D21_PropertyTax_lucknow"
            insight="Zone 1 (Central) shows highest commercial tax collection, indicating prime commercial activity"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={propertyTaxData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis />
                <Tooltip formatter={(value: number) => `₹${value.toFixed(2)} Cr`} />
                <Legend />
                <Bar dataKey="residential" stackId="a" fill="#E6DED2" name="Residential" />
                <Bar dataKey="commercial" stackId="a" fill="#A46B47" name="Commercial" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Governance Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Governance & Civic Participation
          </h2>
          
          <ChartWrapper
            title="Top 20 Wards by Voter Turnout"
            source="D44_Governance_Lucknow"
            insight="Higher voter turnout indicates better civic engagement and potentially better governance"
          >
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={governanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ward" angle={-45} textAnchor="end" height={120} />
                <YAxis />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'turnout' ? `${value}%` : value.toLocaleString()
                } />
                <Legend />
                <Bar dataKey="turnout" fill="#2F3D36" name="Turnout %" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Water & Sanitation Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Water Tax & Sanitation
          </h2>
          
          <ChartWrapper
            title="Water Tax Collection vs Billed by Zone (2017-18)"
            source="D10_WaterTax_Lucknow"
            insight="Collection efficiency shows revenue management quality - higher efficiency indicates better administration"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={waterTaxData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'efficiency' ? `${value}%` : `₹${value.toFixed(2)} Lakhs`
                } />
                <Legend />
                <Bar yAxisId="left" dataKey="collected" fill="#3498db" name="Collected (Lakhs)" />
                <Bar yAxisId="left" dataKey="billed" fill="#e74c3c" name="Billed (Lakhs)" />
                <Bar yAxisId="right" dataKey="efficiency" fill="#2ecc71" name="Efficiency %" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
          
          <ChartWrapper
            title="Water Tax Collection Trends by Zone (2013-2018)"
            source="D10_WaterTax_Lucknow"
            insight="Multi-year trends show which zones have improving or declining revenue collection"
          >
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={waterTaxTrends.slice(0, 8)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip formatter={(value: number) => `₹${value.toFixed(2)} Lakhs`} />
                <Legend />
                <Area type="monotone" dataKey="2013-14" stackId="1" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2014-15" stackId="1" stroke="#3498db" fill="#3498db" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2015-16" stackId="1" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2016-17" stackId="1" stroke="#f39c12" fill="#f39c12" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2017-18" stackId="1" stroke="#9b59b6" fill="#9b59b6" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Sewerage Network Coverage by Zone"
            source="D11_PublicToilet_Lucknow"
            insight="Higher sewerage coverage indicates better infrastructure and quality of life"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={toiletData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'coverage' ? `${value}%` : value.toLocaleString()
                } />
                <Legend />
                <Bar yAxisId="left" dataKey="households" fill="#E6DED2" name="Total Households" />
                <Bar yAxisId="left" dataKey="sewerage" fill="#C2B8A3" name="Sewerage Network" />
                <Bar yAxisId="right" dataKey="coverage" fill="#A46B47" name="Coverage %" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Health & Mortality Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Health & Mortality
          </h2>
          
          <ChartWrapper
            title="Births vs Deaths Trend (2013-18)"
            source="D09_Mortality_Lucknow"
            insight="Natural population growth indicator - positive birth-death ratio indicates healthy population growth"
          >
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={mortalityChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value: number) => `${value.toFixed(1)}K`} />
                <Legend />
                <Area type="monotone" dataKey="births" stackId="1" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.6} name="Live Births (K)" />
                <Area type="monotone" dataKey="deaths" stackId="2" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.6} name="Deaths (K)" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Disease Cases - Vector Borne vs Water Borne"
            source="D46_Diseases_Lucknow"
            insight="Water-borne diseases show significant numbers, indicating need for better water quality management"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={diseasesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value: number) => value.toLocaleString()} />
                <Legend />
                <Bar dataKey="vector" fill="#f39c12" name="Vector Borne" />
                <Bar dataKey="water" fill="#3498db" name="Water Borne" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Transport Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Public Transport
          </h2>
          
          <ChartWrapper
            title="Bus Fleet Composition"
            source="D36_Buses_Lucknow"
            insight="Fleet diversity shows investment in public transport infrastructure"
          >
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={busData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {busData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `${value} buses`} />
              </PieChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Revenue & Economy Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Revenue & Economy
          </h2>
          
          <ChartWrapper
            title="VAT/GST Collection Trend"
            source="D48_VAT_GST_Lucknow"
            insight="Commercial activity indicator - steady growth shows economic development"
          >
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={vatGstData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value: number) => `₹${(value / 1000).toFixed(2)}K`} />
                <Legend />
                <Line type="monotone" dataKey="collection" stroke="#2F3D36" strokeWidth={3} name="Collection (INR)" />
              </LineChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Property Tax Trends Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Property Tax Trends (Multi-Year Analysis)
          </h2>
          
          <ChartWrapper
            title="Residential Property Tax Collection Trends Over Years"
            source="D21_PropertyTax_lucknow"
            insight="Tracking tax collection trends helps identify zones with growing property values and better revenue management"
          >
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={propertyTaxTrendsRaw.slice(0, 8)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip formatter={(value: number) => `₹${value.toFixed(2)} Cr`} />
                <Legend />
                <Area type="monotone" dataKey="2013-14" stackId="1" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2014-15" stackId="1" stroke="#3498db" fill="#3498db" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2015-16" stackId="1" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2016-17" stackId="1" stroke="#f39c12" fill="#f39c12" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2017-18" stackId="1" stroke="#9b59b6" fill="#9b59b6" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Property Tax Collection Efficiency by Zone (2017-18)"
            source="D21_PropertyTax_lucknow"
            insight="Higher efficiency indicates better tax collection and compliance - key for property investment returns"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={propertyTaxEfficiency}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'efficiency' ? `${value}%` : `₹${value.toFixed(2)} Cr`
                } />
                <Legend />
                <Bar yAxisId="left" dataKey="collected" fill="#2ecc71" name="Collected (Cr)" />
                <Bar yAxisId="left" dataKey="demand" fill="#e74c3c" name="Demand (Cr)" />
                <Bar yAxisId="right" dataKey="efficiency" fill="#3498db" name="Efficiency %" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Water Tax Trends Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Water Tax Collection Trends
          </h2>
          
          <ChartWrapper
            title="Water Tax Collection Trends by Zone (2013-2018)"
            source="D10_WaterTax_Lucknow"
            insight="Water tax trends indicate infrastructure quality and service delivery - important for livability"
          >
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip formatter={(value: number) => `₹${value.toFixed(2)} Lakhs`} />
                <Legend />
                <Area type="monotone" dataKey="2013-14" stackId="1" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2014-15" stackId="1" stroke="#3498db" fill="#3498db" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2015-16" stackId="1" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2016-17" stackId="1" stroke="#f39c12" fill="#f39c12" fillOpacity={0.6} />
                <Area type="monotone" dataKey="2017-18" stackId="1" stroke="#9b59b6" fill="#9b59b6" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Digital Payments Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Digital Payments & E-Governance
          </h2>
          
          <ChartWrapper
            title="Digital Payment Adoption Trends"
            source="D50_DigitalPayments_Lucknow"
            insight="Digital payment adoption indicates modern governance and convenience - correlates with higher property values"
          >
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={digitalPaymentsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" angle={-45} textAnchor="end" height={100} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'percentage' ? `${value.toFixed(1)}%` : `₹${value.toFixed(2)} Cr`
                } />
                <Legend />
                <Area yAxisId="left" type="monotone" dataKey="total" stackId="1" stroke="#3498db" fill="#3498db" fillOpacity={0.6} name="Total Collection" />
                <Area yAxisId="left" type="monotone" dataKey="digital" stackId="1" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.8} name="Digital Payment" />
                <Line yAxisId="right" type="monotone" dataKey="percentage" stroke="#e74c3c" strokeWidth={3} name="Digital %" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Road Safety Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Road Safety & Traffic Infrastructure
          </h2>
          
          <ChartWrapper
            title="Road Accidents - Injuries & Fatalities Trend"
            source="D41_Injuries_Fatilities_Lucknow"
            insight="Lower accident rates indicate better road infrastructure and safety - important for property desirability"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={roadSafetyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value: number) => value.toLocaleString()} />
                <Legend />
                <Bar dataKey="injuries" fill="#f39c12" name="Injuries" />
                <Bar dataKey="fatalities" fill="#e74c3c" name="Fatalities" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper
            title="Signalized Intersections by Zone"
            source="D35_Signalized_Intersections_lucknow"
            insight="More operational signalized intersections indicate better traffic management and infrastructure"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={intersectionChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'operationalRate' ? `${value}%` : value.toString()
                } />
                <Legend />
                <Bar yAxisId="left" dataKey="total" fill="#E6DED2" name="Total Intersections" />
                <Bar yAxisId="left" dataKey="operational" fill="#2F3D36" name="Operational" />
                <Bar yAxisId="right" dataKey="operationalRate" fill="#A46B47" name="Operational %" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Top Wards Analysis */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Top Performing Wards
          </h2>
          
          <ChartWrapper
            title="Top 20 Wards by Number of Households"
            source="D03_Households_Lucknow"
            insight="Wards with more households indicate higher population density and potential for property demand"
          >
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={topWards} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="ward" type="category" width={200} />
                <Tooltip formatter={(value: number) => value.toLocaleString()} />
                <Legend />
                <Bar dataKey="households" fill="#2F3D36" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Health Infrastructure by Zone */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Healthcare Infrastructure Distribution
          </h2>
          
          <ChartWrapper
            title="Healthcare Facilities & Bed Capacity by Zone"
            source="D08_HealthInfrastructure_lucknow"
            insight="Better healthcare infrastructure increases property value and livability scores"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={healthZoneData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'facilities' ? `${value} facilities` : `${value} beds`
                } />
                <Legend />
                <Bar yAxisId="left" dataKey="facilities" fill="#3498db" name="No. of Facilities" />
                <Bar yAxisId="right" dataKey="beds" fill="#e74c3c" name="Total Beds" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Waste Management Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Waste Management & Sanitation
          </h2>
          
          <ChartWrapper
            title="Solid Waste Collection Coverage - Top 15 Wards"
            source="D12_D2D_Collection_Coverage"
            insight="100% waste collection coverage indicates better civic services and cleanliness - important for property values"
          >
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={wasteCoverage} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="ward" type="category" width={150} />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'coverage' ? `${value}%` : value.toLocaleString()
                } />
                <Legend />
                <Bar dataKey="coverage" fill="#2ecc71" name="Coverage %" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Open Spaces & Recreation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Open Spaces & Recreation
          </h2>
          
          <ChartWrapper
            title="Open Spaces Distribution by Zone"
            source="D29_openspaces_lucknow"
            insight="More open spaces improve quality of life and increase property values through better livability"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={openSpacesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'count' ? `${value} spaces` : `${value} sq km`
                } />
                <Legend />
                <Bar yAxisId="left" dataKey="count" fill="#2ecc71" name="No. of Spaces" />
                <Bar yAxisId="right" dataKey="area" fill="#27ae60" name="Total Area (sq km)" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Heritage & Culture */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Cultural Heritage Sites
          </h2>
          
          <ChartWrapper
            title="Heritage Sites by Zone"
            source="D31_Cultural_Heritage_lucknow"
            insight="Heritage sites add cultural value and can drive tourism, indirectly benefiting property values"
          >
            <div className="space-y-3">
              {heritageData.map((site: any, idx: number) => (
                <div key={idx} className="border-l-4 border-amber-500 bg-amber-50 p-3 rounded">
                  <div className="font-semibold text-gray-800">{site.name}</div>
                  <div className="text-sm text-gray-600">
                    {site.zone} • {site.nature} • {site.use}
                  </div>
                </div>
              ))}
            </div>
          </ChartWrapper>
        </section>

        {/* Public Transport Efficiency */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
            Public Transport Performance
          </h2>
          
          <ChartWrapper
            title="Bus Revenue vs Cost Efficiency"
            source="D37_EarningsBusTrips_Lucknow"
            insight="Positive revenue-cost ratio indicates sustainable public transport - important for property accessibility"
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={busEfficiencyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value: number, name: string) => 
                  name === 'efficiency' ? `${value.toFixed(1)}%` : `₹${value.toFixed(2)}`
                } />
                <Legend />
                <Bar dataKey="costPerKm" fill="#e74c3c" name="Cost per km" />
                <Bar dataKey="revenuePerKm" fill="#2ecc71" name="Revenue per km" />
                <Bar dataKey="profitPerKm" fill="#3498db" name="Profit per km" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </section>

        {/* Public Amenities Summary */}
        {amenities && (
          <section className="mb-12">
            <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b-2 border-gray-300 pb-2">
              Public Amenities Summary
            </h2>
            
            <ChartWrapper
              title="Key Public Infrastructure Counts"
              source="D05_PublicAmenities_lucknow"
              insight="Comprehensive view of city's public infrastructure facilities"
            >
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Graduation Colleges</div>
                  <div className="text-2xl font-bold text-blue-600">{amenities['No. of Graduation Colleges'] || 'N/A'}</div>
                </div>
                <div className="bg-green-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Public Toilets</div>
                  <div className="text-2xl font-bold text-green-600">{amenities['No. of Public Toilets'] || 'N/A'}</div>
                </div>
                <div className="bg-purple-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Public Hospitals</div>
                  <div className="text-2xl font-bold text-purple-600">{amenities['No. of Public Hospitals'] || 'N/A'}</div>
                </div>
                <div className="bg-orange-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Parks & Gardens</div>
                  <div className="text-2xl font-bold text-orange-600">{amenities['No. of Parks and Gardens'] || 'N/A'}</div>
                </div>
                <div className="bg-red-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Dispensaries</div>
                  <div className="text-2xl font-bold text-red-600">{amenities['No. of Dispensaries / Primary Hospitals'] || 'N/A'}</div>
                </div>
                <div className="bg-indigo-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Libraries</div>
                  <div className="text-2xl font-bold text-indigo-600">{amenities['No. of Libraries'] || 'N/A'}</div>
                </div>
                <div className="bg-teal-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Community Centers</div>
                  <div className="text-2xl font-bold text-teal-600">{amenities['No. of Community Centers'] || 'N/A'}</div>
                </div>
                <div className="bg-pink-50 p-4 rounded">
                  <div className="text-sm text-gray-600">Licensed Hawkers</div>
                  <div className="text-2xl font-bold text-pink-600">{amenities['No. of Licensed Hawkers'] || 'N/A'}</div>
                </div>
              </div>
            </ChartWrapper>
          </section>
        )}
      </div>
    </div>
  );
}

