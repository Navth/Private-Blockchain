import { Building2, Package, Stethoscope } from "lucide-react";
import { randomFromArray, randomNumber } from "../utils/random";
import type { DomainKey, DomainOption, DomainPayload } from "../types/blockchain";

export const DOMAIN_OPTIONS: DomainOption[] = [
  {
    key: "supply-chain",
    label: "Supply Chain",
    description: "Monitoring cold-chain logistics across warehouses and ports.",
    icon: Package,
    accent: "from-blue-500 to-cyan-500",
  },
  {
    key: "healthcare",
    label: "Healthcare",
    description: "Coordinating patient data integrity across hospitals.",
    icon: Stethoscope,
    accent: "from-emerald-500 to-teal-500",
  },
  {
    key: "finance",
    label: "Finance",
    description: "Reconciling inter-bank settlements and approvals.",
    icon: Building2,
    accent: "from-amber-500 to-orange-500",
  },
];

const SUPPLY_CHAIN_LOCATIONS = ["Rotterdam", "Singapore", "Hamburg", "Los Angeles"];
const SUPPLY_CHAIN_PRODUCTS = ["mRNA Vaccines", "Fresh Produce", "Pharmaceuticals", "Microchips"];

const HEALTHCARE_DIAGNOSES = ["Hypertension", "Type II Diabetes", "Asthma", "Cardiac Arrhythmia"];
const HEALTHCARE_TREATMENTS = ["ACE Inhibitors", "Insulin Therapy", "Inhaled Corticosteroids", "Beta Blockers"];

const FINANCE_INSTRUMENTS = ["Letter of Credit", "Syndicated Loan", "Derivatives Bundle", "Treasury Swap"];
const FINANCE_CURRENCIES = ["USD", "EUR", "SGD", "JPY"];

export const buildDomainPayload = (domain: DomainKey): DomainPayload => {
  switch (domain) {
    case "supply-chain":
      return {
        shipmentId: `SC-${randomNumber(1000, 9999)}`,
        product: randomFromArray(SUPPLY_CHAIN_PRODUCTS),
        temperature: `${randomNumber(2, 7)}°C`,
        humidity: `${randomNumber(55, 75)}%`,
        waypoint: randomFromArray(SUPPLY_CHAIN_LOCATIONS),
      };
    case "healthcare":
      return {
        patientId: `HC-${randomNumber(10000, 99999)}`,
        diagnosis: randomFromArray(HEALTHCARE_DIAGNOSES),
        treatmentPlan: randomFromArray(HEALTHCARE_TREATMENTS),
        attendingPhysician: `${randomFromArray(["Dr. Singh", "Dr. Alvarez", "Dr. Carter", "Dr. Huang"])}`,
        followUpDays: randomNumber(7, 28),
      };
    case "finance":
      return {
        transactionRef: `FX-${randomNumber(100000, 999999)}`,
        instrument: randomFromArray(FINANCE_INSTRUMENTS),
        amount: `${randomNumber(250, 900)}M`,
        currency: randomFromArray(FINANCE_CURRENCIES),
        riskScore: randomNumber(1, 5),
      };
    default:
      return {};
  }
};
