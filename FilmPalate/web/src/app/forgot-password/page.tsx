"use client";

import Link from "next/link";
import { useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    setBusy(false);
    // Rate limits are the only error worth showing; otherwise don't reveal whether
    // an account exists for this email.
    if (error && /rate|seconds|too many/i.test(error.message)) {
      setError("Too many reset emails were requested. Please wait a few minutes and try again.");
      return;
    }
    setSent(true);
  }

  return (
    <Card className="mx-auto w-full max-w-sm">
      <CardHeader>
        <CardTitle>Reset your password</CardTitle>
        <CardDescription>Enter your account email and we&apos;ll send you a link to choose a new password.</CardDescription>
      </CardHeader>
      <CardContent>
        {sent ? (
          <div className="flex flex-col gap-4">
            <Alert>
              <AlertDescription>
                If there&apos;s an account for <strong>{email.trim()}</strong>, a reset link is on its way. Open it on this
                same device and browser. It can take a minute, so check your spam folder too.
              </AlertDescription>
            </Alert>
            <Button variant="outline" onClick={() => setSent(false)}>Send another link</Button>
          </div>
        ) : (
          <form onSubmit={submit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" autoComplete="email" required value={email}
                     onChange={(e) => setEmail(e.target.value)} />
            </div>
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            <Button type="submit" disabled={busy}>{busy ? "Sending…" : "Send reset link"}</Button>
          </form>
        )}
        <p className="mt-4 text-center text-sm text-muted-foreground">
          Remembered it? <Link className="underline" href="/login">Log in</Link>
        </p>
      </CardContent>
    </Card>
  );
}
