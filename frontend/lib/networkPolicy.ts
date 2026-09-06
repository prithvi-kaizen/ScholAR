"use client";

import { useEffect, useState } from "react";
import { isNetworkPolicyStatus, type NetworkPolicyStatus } from "../types/api";

export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export function useNetworkPolicy(): {
  policy: NetworkPolicyStatus | null;
  loading: boolean;
  error: string | null;
} {
  const [policy, setPolicy] = useState<NetworkPolicyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${BACKEND_URL}/api/system/network-policy`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Network policy endpoint returned HTTP ${response.status}`);
        const payload: unknown = await response.json();
        if (!isNetworkPolicyStatus(payload)) throw new Error("Network policy response failed validation");
        setPolicy(payload);
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Network policy is unavailable");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  return { policy, loading, error };
}
