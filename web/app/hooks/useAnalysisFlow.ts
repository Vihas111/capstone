"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  analysisErrorMessage,
  fetchDrugs,
  isSuccessfulAnalysis,
  predict,
  PredictResponse,
} from "../lib/api";

export type AnalysisPhase = "landing" | "investigating" | "report" | "error";

function normalize(input: string): string {
  return input.trim().toLowerCase();
}

export function useAnalysisFlow() {
  const [phase, setPhase] = useState<AnalysisPhase>("landing");
  const [drugs, setDrugs] = useState<string[]>([]);
  const [drugCount, setDrugCount] = useState<2 | 3>(2);
  const [drugA, setDrugA] = useState("");
  const [drugB, setDrugB] = useState("");
  const [drugC, setDrugC] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [bootstrapLoading, setBootstrapLoading] = useState(true);
  const [investigationDone, setInvestigationDone] = useState(false);

  const analyzingRef = useRef(false);
  const requestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const loadDrugs = useCallback(async () => {
    setBootstrapLoading(true);
    setBootstrapError(null);
    try {
      const list = await fetchDrugs();
      setDrugs(list);
    } catch (err: unknown) {
      setDrugs([]);
      const message =
        err instanceof Error ? err.message : "Failed to load drug records.";
      setBootstrapError(message);
    } finally {
      setBootstrapLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDrugs();
  }, [loadDrugs]);

  const validDrugSet = useMemo(() => new Set(drugs.map(normalize)), [drugs]);

  const setDrugCountSafe = (count: 2 | 3) => {
    setDrugCount(count);
    if (count === 2) setDrugC("");
  };

  const canAnalyze =
    !bootstrapLoading &&
    !bootstrapError &&
    validDrugSet.has(normalize(drugA)) &&
    validDrugSet.has(normalize(drugB)) &&
    (drugCount === 2 || validDrugSet.has(normalize(drugC)));

  const analyze = async () => {
    if (!canAnalyze || analyzingRef.current) return;

    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    analyzingRef.current = true;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("investigating");
    setInvestigationDone(false);
    setResult(null);
    setError(null);

    const drugList =
      drugCount === 3
        ? [drugA.trim(), drugB.trim(), drugC.trim()]
        : [drugA.trim(), drugB.trim()];

    try {
      const response = await predict(drugList, {
        signal: controller.signal,
      });

      if (requestIdRef.current !== requestId) return;

      if (!isSuccessfulAnalysis(response)) {
        setError(analysisErrorMessage(response));
        setPhase("error");
        return;
      }

      setResult(response);
      setInvestigationDone(true);
    } catch (err: unknown) {
      if (requestIdRef.current !== requestId) return;
      const message = err instanceof Error ? err.message : "Connection failed.";
      setError(message);
      setPhase("error");
    } finally {
      if (requestIdRef.current === requestId) {
        analyzingRef.current = false;
        abortRef.current = null;
      }
    }
  };

  const cancel = () => {
    requestIdRef.current += 1;
    analyzingRef.current = false;
    abortRef.current?.abort();
    abortRef.current = null;
    setInvestigationDone(false);
    setPhase("landing");
    setResult(null);
    setError(null);
  };

  const reset = () => {
    cancel();
  };

  const completeInvestigation = useCallback(() => {
    setPhase("report");
  }, []);

  return {
    phase,
    drugs,
    drugCount,
    setDrugCount: setDrugCountSafe,
    drugA,
    drugB,
    drugC,
    setDrugA,
    setDrugB,
    setDrugC,
    canAnalyze,
    bootstrapError,
    bootstrapLoading,
    investigationDone,
    result,
    error,
    analyze,
    cancel,
    reset,
    retryBootstrap: loadDrugs,
    completeInvestigation,
  };
}
