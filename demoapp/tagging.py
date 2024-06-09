category_for_bill = """
[Environmental, Education, Health Care, Criminal Justice, Taxation, Transportation, 
Housing, Civil Rights, Labor and Employment, Budget and Appropriations, Public Safety, Technology and Innovation, 
Immigration, Economic Development, Social Services]
"""

tags_for_bill = """
{
    "Environmental": [
        "Climate Change Mitigation",
        "Renewable Energy Initiatives",
        "Biodiversity Conservation",
        "Water Pollution Control",
        "Sustainable Agriculture Practices",
        "Air Quality Improvement",
        "Waste Reduction Programs",
        "Coastal Erosion Management",
        "Environmental Impact Assessments",
        "Wildlife Habitat Protection",
        "Green Infrastructure Development",
        "Eco-Friendly Urban Planning",
        "Other"
    ], 

    "Education": [
        "Curriculum Development",
        "Digital Learning Resources",
        "Teacher Professional Development",
        "Early Childhood Education",
        "Special Education Services",
        "Literacy Programs",
        "Vocational Training",
        "Education Technology Integration",
        "School Infrastructure Upgrades",
        "Student Mental Health Support",
        "Educational Equity Policies",
        "Higher Education Accessibility",
        "Other"
    ],

    "Health Care": [
        "Universal Health Coverage",
        "Healthcare Access for All",
        "Mental Health Services Expansion",
        "Disease Prevention Programs",
        "Elderly Care Services",
        "Healthcare Quality Standards",
        "Health Information Technology",
        "Maternal and Child Health",
        "Public Health Emergency Preparedness",
        "Healthcare Workforce Training",
        "Telemedicine Services",
        "Drug Pricing and Regulation"
        "Other"
    ],

    "Criminal Justice": [
        "Police Reforms",
        "Prisoner Rehabilitation",
        "Community Policing Initiatives",
        "Restorative Justice Programs",
        "Criminal Sentencing Reform",
        "Victim Support Services",
        "Legal Aid for the Underprivileged",
        "Juvenile Justice System Overhaul",
        "Hate Crime Prevention Measures",
        "Court System Modernization",
        "Decriminalization Policies",
        "Bail Reform"
        "Other"
    ],

    "Taxation": [
        "Progressive Tax Reform",
        "Corporate Taxation Policies",
        "Income Tax Deductions",
        "Property Tax Relief",
        "Sales Tax Revision",
        "Tax Compliance Regulations",
        "Tax Transparency Measures",
        "Wealth Redistribution Initiatives",
        "Small Business Tax Breaks",
        "Tax Fraud Prevention Measures",
        "Digital Economy Taxation",
        "Excise Taxes and Tariffs"
        "Other"
    ],

    "Transportation": [
        "Public Transit Expansion",
        "Road Infrastructure Maintenance",
        "Bike and Pedestrian Path Development",
        "Traffic Congestion Reduction",
        "Electric Vehicle Adoption Incentives",
        "Freight Transportation Efficiency",
        "Aviation Safety Regulations",
        "Railroad Infrastructure Modernization",
        "Intermodal Transportation Integration",
        "Autonomous Vehicle Regulations",
        "Transportation Emission Controls",
        "Public Transportation Accessibility"
        "Other"
    ],

    "Housing": [
        "Affordable Housing Development",
        "Homelessness Prevention Programs",
        "Rent Control Measures",
        "Fair Housing Enforcement",
        "Housing Discrimination Prevention",
        "Housing Voucher Program Expansion",
        "Sustainable Housing Initiatives",
        "Urban Redevelopment Plans",
        "Eviction Moratoriums",
        "Homeownership Support Programs",
        "Tenant Rights and Protections",
        "Rural Housing Development"
        "Other"
    ],

    "Civil Rights": [
        "Voting Rights Expansion",
        "Government Transparency and Accountability",
        "Privacy and Data Protection",
        "Freedom of Speech and Expression",
        "Anti-Discrimination Laws",
        "Gender Equality Protections",
        "LGBTQ+ Rights Advocacy",
        "Disability Rights and Accessibility",
        "Religious Freedom",
        "Minority Rights and Protections",
        "Citizens' Rights in Digital Age",
        "Access to Public Services"
        "Other"
    ],

    "Labor and Employment": [
        "Minimum Wage Increase",
        "Occupational Safety Regulations",
        "Labor Union Protections",
        "Workplace Harassment Prevention",
        "Job Training and Apprenticeships",
        "Workforce Diversity Initiatives",
        "Employee Benefits Expansion",
        "Fair Working Hours Regulations",
        "Unemployment Benefits Enhancement",
        "Remote Work Policies",
        "Workers' Compensation Reforms",
        "Employment Discrimination Laws"
        "Other"
    ],

    "Budget and Appropriations": [
        "Government Spending Oversight",
        "Emergency Fund Allocation",
        "Public Debt Management",
        "Fiscal Responsibility Audits",
        "Public Infrastructure Investment",
        "Social Welfare Program Funding",
        "Pension Plan Reform",
        "Local Government Grant Programs",
        "Financial Aid for Disadvantaged Communities",
        "Tax Revenue Allocation",
        "Budget Transparency",
        "Fiscal Impact Assessments"
        "Other"
    ],

    "Public Safety": [
        "Emergency Response Planning",
        "Disaster Preparedness Training",
        "Cybersecurity Protocols",
        "Domestic Violence Prevention",
        "Fire Safety Regulations",
        "Gun Control Measures",
        "Community Health and Safety Programs",
        "Public Health Crisis Management",
        "Hate Crime Reporting Systems",
        "Crime Prevention Initiatives",
        "First Responder Support",
        "Public Safety Technology"
        "Other"
    ],

    "Technology and Innovation": [
        "Digital Privacy Laws",
        "Data Security Measures",
        "Innovation Investment Policies",
        "Broadband Infrastructure Expansion",
        "E-Government Service Enhancements",
        "Technology Education in Schools",
        "Artificial Intelligence Regulations",
        "Privacy Protection for Biometric Data",
        "Blockchain Integration Strategies",
        "Open Data Initiatives",
        "Tech Start-up Support",
        "Digital Infrastructure Security"
        "Other"
    ],

    "Immigration": [
        "Immigration Policy Reform",
        "Refugee Resettlement Programs",
        "Asylum Seeker Protections",
        "Migrant Worker Rights",
        "Language Access Services",
        "Family Reunification Initiatives",
        "Pathways to Citizenship",
        "Humanitarian Aid for Migrants",
        "Border Security Measures",
        "Anti-Trafficking Efforts",
        "Visa Policy and Regulations",
        "Immigrant Integration Programs"
        "Other"
    ],

    "Economic Development": [
        "Small Business Support Programs",
        "Export Promotion Policies",
        "Rural Development Initiatives",
        "Entrepreneurship Training",
        "Trade Agreement Negotiations",
        "Tourism Industry Growth",
        "Regional Economic Integration",
        "Financial Inclusion Programs",
        "Economic Diversification Strategies",
        "Infrastructure Investment Plans",
        "Economic Impact of Climate Change",
        "Innovation Hubs and Clusters"
        "Other"
    ],

    "Social Services": [
        "Child Welfare Services",
        "Domestic Violence Support",
        "Elderly Care Programs",
        "Foster Care System Reforms",
        "Disability Assistance Programs",
        "Community Support Services",
        "Youth Mentorship Programs",
        "Substance Abuse Rehabilitation",
        "Home Care for the Disabled",
        "Affordable Childcare Services",
        "Mental Health Services Accessibility",
        "Social Inclusion Initiatives"
        "Other"
    ]
}
"""


new_categories_for_bill_list = ['Commerce', 'Crime and Law Enforcement', 'Economics and Public Finance', 'Education', 'Emergency Management', 'Energy', 'Environmental Protection', 
'Families', 'Government Operations and Politics', 'Health and Food', 'Housing and Community Development', 'Immigration', 'Labor and Employment', 'Law', 'Public and Natural Resources', 
'Science, Technology, Communications', 'Social Sciences and History', 'Social Services', 'Sports and Recreation', 'Taxation', 'Transportation and Public Works']

new_tags_for_bill_dict = {'Commerce': ['Banking and financial institutions regulation',
          'Business ethics',
          'Competition and antitrust',
          'Consumer affairs',
          'Corporate finance and management',
          'Marketing and advertising',
          'Retail and wholesale trades',
          'Securities'],
         'Crime and Law Enforcement': ['Assault and harassment offenses',
          'Crimes against animals and natural resources',
          'Crimes against children',
          'Crimes against property',
          'Criminal investigation, prosecution, interrogation',
          'Criminal justice information and records',
          'Criminal procedure and sentencing',
          'Firearms and explosives',
          'Fraud offenses and financial crimes',
          'Correctional Facilities ',
          'Criminal Justice Reform '],
         'Economics and Public Finance': ['Budget deficits and national debt',
          'Budget process',
          'Business expenses',
          'Currency',
          'Debt collection',
          'Economic development',
          'Economic performance and conditions',
          'Economic theory',
          'Employment taxes',
          'Finance and Financial Sector',
          'Financial crises and stabilization',
          'Financial literacy',
          'Financial services and investments',
          'Inflation and prices',
          'Interest, dividends, interest rates',
          'Labor-management relations',
          'Pension and retirement benefits'],
         'Education': ['Academic performance and assessments',
          'Adult education and literacy',
          'Educational facilities and institutions',
          'Elementary and secondary education',
          'Higher education',
          'Special education',
          'Student aid and college costs',
          'Teaching, teachers, curricula',
          'Technology assessment',
          'Technology transfer and commercialization',
          'Vocational and technical education'],
         'Emergency Management': ['Accidents',
          'Disaster relief and insurance',
          'Emergency communications systems',
          'Emergency medical services and trauma care',
          'Emergency planning and evacuation',
          'Hazards and emergency operations',
          'Search and rescue operations'],
         'Energy': ['Energy assistance for the poor and aged',
          'Energy efficiency and conservation',
          'Energy prices',
          'Energy research',
          'Energy revenues and royalties',
          'Energy storage, supplies, demand',
          'Renewable energy sources'],
         'Environmental Protection': ['Air quality',
          'Environmental assessment, monitoring, research',
          'Environmental education',
          'Environmental health',
          'Environmental regulatory procedures',
          'Hazardous wastes and toxic substances',
          'Pollution control and abatement',
          'Soil pollution',
          'Solid waste and recycling',
          'Water quality',
          'Wetlands'],
         'Families': ['Adoption and foster care',
          'Family planning and birth control',
          'Family relationships',
          'Family services',
          'Marriage and family status',
          'Parenting'],
         'Government Operations and Politics': ['Census and government statistics',
          'Election administration',
          'Government ethics and transparency',
          'Government information and archives',
          'Government studies and investigations',
          'Government trust funds',
          'Legislative rules and procedure',
          'Lobbying and campaign finance',
          'Political advertising',
          'Political parties and affiliation',
          'Political representation',
          'Public contracts and procurement',
          'Public participation and lobbying',
          'public-private cooperation'],
         'Health and Food': ['Alcoholic beverages',
          'Allergies',
          'Alternative treatments',
          'Cancer',
          'Cardiovascular and respiratory health',
          'Dental care',
          'Digestive and metabolic diseases',
          'Drug safety, medical device, and laboratory regulation',
          'Drug therapy',
          'Drug, alcohol, tobacco use',
          'Endangered and threatened species',
          'Food industry and services',
          'Food supply, safety, and labeling',
          'Health care costs and insurance',
          'Health care coverage and access',
          'Health care quality',
          'Health facilities and institutions',
          'Health information and medical records',
          'Health personnel',
          'Health programs administration and funding',
          'Health promotion and preventive care',
          'Health technology, devices, supplies',
          'Hearing, speech, and vision care',
          'Hereditary and development disorders',
          'HIV/AIDS',
          'Medical education',
          'Medical ethics',
          'Medical research',
          'Medical tests and diagnostic methods',
          'Mental health',
          'Musculoskeletal and skin diseases',
          'Neurological disorders',
          'Nutrition and diet',
          'Prescription drugs',
          'Public health',
          'Radiation',
          'Sex and reproductive health'],
         'Housing and Community Development': ['Community life and organization',
          'Commuting',
          'Cooperative and condominium housing',
          'Homelessness and emergency shelter',
          'Housing discrimination',
          'Housing finance and home ownership',
          'Housing for the elderly and disabled',
          'Housing industry and standards',
          'Housing supply and affordability',
          'Landlord and tenant',
          'Low- and moderate-income housing',
          'Residential rehabilitation and home repair'],
         'Immigration': ['Border security and unlawful immigration',
          'Citizenship and naturalization',
          'Immigrant health and welfare',
          'Immigration status and procedures',
          'Refugees, asylum, displaced persons',
          'Visa and passport requirements'],
         'Labor and Employment': ['Employee benefits and pensions',
          'Employee hiring',
          'Employee leave',
          'Employee performance',
          'Employment and training programs',
          'Employment discrimination and employee rights',
          'Labor market',
          'Labor standards',
          'Migrant, seasonal, agricultural labor',
          'Self-employed',
          'Temporary and part-time employment',
          "Workers' compensation",
          'Worker safety and health',
          'Youth employment and child labor'],
         'Law': ['Administrative law and regulatory procedures',
          'Administrative remedies',
          'Civil actions and liability',
          'Civil disturbances',
          'Evidence and witnesses',
          'Judicial procedure and administration',
          'Judicial review and appeals',
          'Jurisdiction and venue',
          'Legal fees and court costs',
          'Property rights',
          'Rule of law and government transparency'],
         'Public and Natural Resources': ['Forests, forestry, trees',
          'General public lands matters',
          'Marine and coastal resources, fisheries',
          'Marine pollution',
          'Monuments and memorials',
          'Water resources funding',
          'Wilderness'],
         'Science, Technology, Communications': ['Advanced technology and technological innovations',
          'Atmospheric science and weather',
          'Computer security and identity theft',
          'Computers and information technology',
          'Earth sciences',
          'Ecology',
          'Environmental technology',
          'Genetics',
          'Internet, web applications, social media',
          'Photography and imaging',
          'Radio spectrum allocation',
          'Telecommunication rates and fees',
          'Telephone and wireless communication',
          'Television and film'],
         'Social Sciences and History': ['Area studies and international education',
          'Archaeology and anthropology',
          'History and historiography',
          'Language arts',
          'Policy sciences',
          'World history',
          'Food assistance and relief'],
         'Social Services': ['Child care and development',
          'Domestic violence and child abuse',
          'Home and outpatient care',
          'Social work, volunteer service, charitable organizations',
          'Unemployment',
          'Urban and suburban affairs and development',
          "Veterans' education, employment, rehabilitation",
          "Veterans' loans, housing, homeless programs",
          "Veterans' medical care"],
         'Sports and Recreation': ['Athletes',
          'Games and hobbies',
          'Hunting and fishing',
          'Outdoor recreation',
          'Parks, recreation areas, trails',
          'Performing arts',
          'Professional sports',
          'Sports and recreation facilities'],
         'Taxation': ['Capital gains tax',
          'Corporate tax',
          'Estate tax',
          'Excise tax',
          'Gift tax',
          'Income tax',
          'Inheritance tax',
          'Payroll tax',
          'Property tax',
          'Sales tax',
          'Tariffs',
          'Transfer and inheritance taxes',
          'Tax-exempt organizations'],
         'Transportation and Public Works': ['Aviation and airports',
          'Highways and roads',
          'Maritime affairs and fisheries',
          'Mass transit and transportation',
          'Public utilities and utility rates',
          'Railroads',
          'Transportation safety and security',
          'Water storage',
          'Water use and supply']}